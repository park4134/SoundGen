import sys
sys.path.append('./')
sys.path.append('./utils/')

import os
import ast
import json
import torch
import random
import argparse

import numpy as np

from glob import glob
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import GreatestHitsDatasetDummy
from models.stage2.model import Stage1DummyModel
from stable_audio_controlnet.main.controlnet.pretrained import get_pretrained_controlnet_model

class Trainer():
    def __init__(self):
        self.device = torch.device("cuda:1")
        self._set_seed()
        self._get_args()
        self._get_paths()
        self._init_dataloader()
        self._init_model()

    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default="preprocessed_15_5.0_(112, 112)_48000")
        parser.add_argument('--timesteps', type=int, default=100)
        parser.add_argument('--batch', type=int, default=16)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--patience', type=int, default=10)
        self.args = parser.parse_args()
    
    def _get_paths(self):
        self.save_path = os.path.join(os.getcwd(), 'results', 'stage2', 'train')
        os.makedirs(self.save_path, exist_ok=True)
        num = len(glob(os.path.join(self.save_path, 'train_*')))
        self.save_dir = f'train_{num}'
        self.log_dir = os.path.join(self.save_path, self.save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_json_path = os.path.join(self.save_path, self.save_dir, 'train_log.json')
    
    def _init_dataloader(self):
        data_info = self.args.data_dir.split('_')
        image_fps = int(data_info[1])
        chunk_length_sec = float(data_info[2])
        image_size = ast.literal_eval(data_info[3])

        train_dataset = GreatestHitsDatasetDummy(
            root_dir=os.path.join(os.getcwd(), 'data', 'greatest_hits', self.args.data_dir),
            split_file_path="./data/greatest_hits/train.txt",
            chunk_length_sec=chunk_length_sec,
            image_fps=image_fps,
            image_size=image_size
        )
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True)

        val_dataset = GreatestHitsDatasetDummy(
            root_dir=os.path.join(os.getcwd(), 'data', 'greatest_hits', self.args.data_dir),
            split_file_path="./data/greatest_hits/val.txt",
            chunk_length_sec=chunk_length_sec,
            image_fps=image_fps,
            image_size=image_size
        )

        self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch, shuffle=False)
    
    def _init_model(self):
        self.model_stg1 = Stage1DummyModel()
        self.model_stg2, self.model_stg2_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0", controlnet_types=["envelope"], depth_factor=0.2)

        self.model_stg2.model.model.requires_grad_(False)
        self.model_stg2.conditioner.requires_grad_(False)
        self.model_stg2.conditioner.eval()
        self.model_stg2.pretransform.requires_grad_(False)
        self.model_stg2.pretransform.eval()

        self.model_stg1.to(self.device)
        self.model_stg2.to(self.device)
    
    def q_sample(self, x_start, t, noise, alphas_cumprod):
        """
        x_start: clean audio (B, 1, L)
        t: timestep tensor (B,)
        noise: same shape as x_start
        alphas_cumprod: (T,)
        """
        # gather alpha_t for each t
        sqrt_alpha = alphas_cumprod[t].view(-1, 1, 1).sqrt()
        sqrt_one_minus_alpha = (1.0 - alphas_cumprod[t]).view(-1, 1, 1).sqrt()
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def validate(self, epoch):
        self.model_stg1.eval()
        self.model_stg2.eval()

        total_loss = 0.0
        num_batches = 0

        loop = tqdm(self.val_dataloader, desc=f"[Epoch {epoch+1}/{self.args.epochs}]")

        with torch.no_grad():
            for batch in loop:
                frames = batch["frames"].to(self.device)
                audio = batch["waveform"].to(self.device)
                rms = batch["rms"]
                prompt = batch["prompt"]
                start_time = batch["start_time"]
                end_time = batch["end_time"]
                
                # Get image feature & RMS envelope from Stage1 model
                image_feat = self.model_stg1(frames)
                x0 = self.model_stg2.pretransform.encode(audio.repeat(1, 2, 1))

                # Generate random noised audio
                noise = torch.randn_like(x0)
                B = audio.shape[0]
                t = torch.randint(0, self.args.timesteps, (B,), device=self.device)  # (B,)
                noised_latent = self.q_sample(x0, t, noise, self.alphas_cumprod)  # (B, 1, L)

                # Conditioning for ControlNet
                conditioning = [{
                    "envelope": rms[i:i+1].to(self.device), # (1, 1, L)
                    "image_feature": image_feat[i:i+1].unsqueeze(0).to(self.device), # (1, 1, 512)
                    "prompt": prompt[i],
                    "seconds_start": start_time[i],
                    "seconds_total": end_time[i]
                } for i in range(frames.shape[0])]

                # model_stg2가 loss를 반환하는 형태라고 가정
                pred_noise = self.model_stg2(
                    x=noised_latent,
                    t=t.to(self.device),
                    cond=self.model_stg2.conditioner(conditioning, device=self.device),
                    cfg_dropout_prob=0.2,)

                loss = torch.nn.functional.mse_loss(pred_noise, noise)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def train(self):
        # Noise configuration for diffusion model
        betas = torch.linspace(1e-4, 0.02, self.args.timesteps).to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

        optimizer = torch.optim.Adam(list(self.model_stg2.model.controlnet.parameters()), self.args.lr)
        best_loss = float('inf')
        patience = self.args.patience
        no_improve = 0

        # Logging
        writer = SummaryWriter(log_dir=self.log_dir)
        log_data = {"train": [], "val": []}

        for epoch in range(self.args.epochs):
            self.model_stg1.eval()  # stage1은 고정
            self.model_stg2.train()  # stage2의 controlnet만 학습

            total_loss = 0.0

            loop = tqdm(self.train_dataloader, desc=f"[Epoch {epoch+1}/{self.args.epochs}]")

            for batch in loop:
                frames = batch["frames"].to(self.device)
                audio = batch["waveform"].to(self.device)
                rms = batch["rms"]
                prompt = batch["prompt"]
                start_time = batch["start_time"]
                end_time = batch["end_time"]

                # Get image feature & RMS envelope from Stage1 model
                with torch.no_grad():
                    image_feat = self.model_stg1(frames)
                    x0 = self.model_stg2.pretransform.encode(audio.repeat(1, 2, 1))
                
                # Generate random noised audio
                noise = torch.randn_like(x0)
                B = audio.shape[0]
                t = torch.randint(0, self.args.timesteps, (B,), device=self.device)  # (B,)
                noised_latent = self.q_sample(x0, t, noise, self.alphas_cumprod)  # (B, 1, L)

                # Conditioning for ControlNet
                conditioning = [{
                    "envelope": rms[i:i+1].to(self.device), # (1, 1, L)
                    "image_feature": image_feat[i:i+1].unsqueeze(0).to(self.device), # (1, 1, 512)
                    "prompt": prompt[i],
                    "seconds_start": start_time[i],
                    "seconds_total": end_time[i]
                } for i in range(frames.shape[0])]

                # model_stg2가 loss를 반환하는 형태라고 가정
                pred_noise = self.model_stg2(
                    x=noised_latent,
                    t=t.to(self.device),
                    cond=self.model_stg2.conditioner(conditioning, device=self.device),
                    cfg_dropout_prob=0.2,)
                    # device=self.device)

                loss = torch.nn.functional.mse_loss(pred_noise, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

            val_loss = self.validate(epoch)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}")

            # TensorBoard logging
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)

            # JSON 로그 업데이트
            log_data["train"].append({"epoch": epoch + 1, "loss": avg_train_loss})
            log_data["val"].append({"epoch": epoch + 1, "loss": val_loss})

            with open(self.log_json_path, "w") as f:
                json.dump(log_data, f, indent=2)

            # Early stopping & checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                ckpt_path = os.path.join(self.save_path, self.save_dir, 'best_model.pt')
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(self.model_stg2.model.controlnet.state_dict(), ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        writer.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()