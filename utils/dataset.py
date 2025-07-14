import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from opticalflow import extract_optical_flow_farneback_drumstick
import os
from glob import glob
from natsort import natsorted
from PIL import Image
import numpy as np
import json
import torchaudio
import librosa
from preprocess.data_utils import RMS
from torch.utils.data import DataLoader

class GreatestHitsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split_file_path,
        chunk_length_sec=5.0,
        image_fps = 15,
        image_size=(112,112),
        audio_file_suffix="resampled.wav",
        onset_file_suffix="onset.npy",
        metadata_file_suffix="metadata.json",
        rms_file_suffix="rms.npy",
        frame_file_suffix=".png",
        transform_=None,
        use_rms=True,        
        n_rms_classes=64,
        mu_rms=255,
        rms_num_bins = 64,
        rms_min = 0.01
    ):
        super().__init__()
        self.root_dif = root_dir
        self.chunk_length_sec = chunk_length_sec
        self.audio_file_suffix = audio_file_suffix
        self.onset_file_suffix = onset_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.rms_file_suffix = rms_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.use_rms = use_rms
        self.n_rms_classes = n_rms_classes
        self.mu_rms = mu_rms
        self.rms_num_bins = rms_num_bins
        self.rms_min = rms_min
        self.target_frame_len = int(self.chunk_length_sec * image_fps)
        self.flow_hsv_lower = np.array([0, 0, 0])
        self.flow_hsv_upper = np.array([255, 255, 255])


        
        if transform_ is not None:
            self.transform = transform_
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        with open(split_file_path, "r") as f:
            self.video_names = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        
        for video_name in self.video_names:
        # for video_name in self.video_names[:4]:
            video_base_dir = os.path.join(root_dir, video_name)
            chunk_dirs = natsorted(glob(os.path.join(video_base_dir, "chunk_*")))

            for chunk_dir in chunk_dirs:
                # chunk_subdir = os.path.join(chunk_dir, video_name)
                # if not os.path.isdir(chunk_subdir):
                #     continue
                frames_dir = os.path.join(chunk_dir, "frames")
                frames = natsorted(glob(os.path.join(frames_dir, f"*{self.frame_file_suffix}")))
                if not frames:
                    continue
                    
                onset_path = os.path.join(chunk_dir, self.onset_file_suffix)
                meta_path = os.path.join(chunk_dir, self.metadata_file_suffix)
                rms_path = os.path.join(chunk_dir, self.rms_file_suffix)
                audio_path = os.path.join(chunk_dir, self.audio_file_suffix)
                
                self.samples.append({
                    "video_name": video_name,
                    "chunk_dir": chunk_dir,
                    "frames": frames,
                    "onset_path": onset_path,
                    "meta_path": meta_path,
                    "rms_path": rms_path,
                    "audio_path": audio_path,
                })
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []
        for img_path in sample["frames"]:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            

        if len(imgs) > self.target_frame_len:
            imgs = imgs[:self.target_frame_len]
        # elif len(imgs) < self.target_frame_len:
        #     pad = self.target_frame_len - len(imgs)
        #     pad_tensor = torch.zeros_like(imgs[0])
        #     imgs += [pad_tensor] * pad            
            
            
        imgs = torch.stack(imgs, dim=0) #(T, C, H, W)
        
        # Optical Flow 계산
        flow_list, _, _ = extract_optical_flow_farneback_drumstick(
            imgs, mode="hsv", lower=self.flow_hsv_lower, upper=self.flow_hsv_upper
        )
        flow_np = np.stack(flow_list, axis=0)  # (T-1, H, W, 2)
        zero_pad = np.zeros_like(flow_np[0:1])  # (1, H, W, 2)
        flow_np = np.concatenate([zero_pad, flow_np], axis=0)  # (T, H, W, 2)
        flow_tensor = torch.from_numpy(flow_np).permute(0, 3, 1, 2).float()  # (T, 2, H, W)
        
        onset_time_array = np.zeros(10, dtype=np.float32)
        
        
        onset = None
        if os.path.exists(sample["onset_path"]):
            onset = np.load(sample["onset_path"])

        rms = None
        if self.use_rms and os.path.exists(sample["rms_path"]):
            rms = np.load(sample["rms_path"])

            
            
        if onset is not None:
            onset_indices = np.where(onset > 0)[0]
            # print('len onset',len(onset))
            num_onsets = min(10, len(onset_indices))
            
            for i in range(num_onsets):
                idx = onset_indices[i]
                time_sec = idx / 240000
                onset_time_array[i] = time_sec


        # rms 분류 레이블 생성
        if self.use_rms and os.path.exists(sample["rms_path"]):
            
            # µ-law 기반 클래스 이산화
            rms_tensor = torch.tensor(rms, dtype=torch.float32)
            mu_bins = RMS.get_mu_bins(
                mu=self.mu_rms,
                num_bins=self.rms_num_bins,
                rms_min=self.rms_min
            )
            rms_classes = RMS.discretize_rms(rms_tensor, mu_bins).long()



        text, start_time, end_time = "", 0.0, 0.0
        if os.path.exists(sample["meta_path"]):
            with open(sample["meta_path"], "r") as f:
                meta = json.load(f)
            text = meta.get("prompt", "")
            start_time = meta.get("seconds_start", 0.0)
            end_time = meta.get("seconds_total", 0.0)

        wav = None
        if os.path.exists(sample["audio_path"]):
            wav, sr = torchaudio.load(sample["audio_path"])
        else:
            sr = None

        return {
            "video_name": sample["video_name"],
            "frames": imgs,
            "flow" : flow_tensor,
            "onset_times": torch.tensor(onset_time_array), #
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "rms": torch.tensor(rms), #
            'rms_class': rms_classes,
            "waveform": wav,
            "audio_sr": sr,
        }
        

class GreatestHitsDatasetDummy(GreatestHitsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- 이미지 프레임 불러오기 ---
        imgs = []
        for img_path in sample["frames"]:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)

        if len(imgs) > self.target_frame_len:
            imgs = imgs[:self.target_frame_len]

        imgs = torch.stack(imgs, dim=0)       # (T, C, H, W)
        imgs = imgs.permute(1, 0, 2, 3)       # (C, T, H, W)

        # --- 오디오 불러오기 ---
        wav = None
        if os.path.exists(sample["audio_path"]):
            wav, sr = torchaudio.load(sample["audio_path"])
            # wav = wav.unsqueeze(0) # shape: (1, 1, L)

        # --- RMS 불러오기 ---
        rms = None
        if self.use_rms and os.path.exists(sample["rms_path"]):
            rms = np.load(sample["rms_path"])
            rms = torch.tensor(rms, dtype=torch.float32)
            rms = rms.unsqueeze(0).unsqueeze(0) # shape: (1, 1, 1875)
        
        rms = F.interpolate(rms, size=240000, mode='linear', align_corners=False) # shape: (1, 1, L)
        rms = rms.squeeze(0)
        
        if os.path.exists(sample["meta_path"]):
            with open(sample["meta_path"], "r") as f:
                meta = json.load(f)
            prompt = meta.get("prompt", "")
            start_time = meta.get("seconds_start", 0.0)
            end_time = meta.get("seconds_total", 0.0)

        return {
            "frames": imgs,      # (C, T, H, W)
            "waveform": wav,     # (1, 1, L)
            "rms": rms,          # (1, 1, L)
            "prompt": prompt,
            "start_time": start_time,
            "end_time": end_time
        }

class FourierFeatures(nn.Module):
    def __init__(self, in_features=1, out_features=12, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


if __name__ == "__main__":
    # dataset = GreatestHitsDataset(
    dataset = GreatestHitsDatasetDummy(
        root_dir="./data/greatest_hits/preprocessed_15_5.0_(112, 112)_48000",
        split_file_path="./data/greatest_hits/train.txt",
        chunk_length_sec=5.0,
        image_size=(112, 112),
    )
    
    # print("총 chunk 샘플:", len(dataset))
    # item = dataset[10]
    # print("frames:", item["frames"].shape)
    # print("rms:", item["rms"].shape)
    # print("waveform:", item["waveform"].shape if item["waveform"] is not None else None)
    # print("prompt:", item["prompt"])
    # print("start_time:", item["start_time"])
    # print("end_time:", item["end_time"])
    
    dataloader = DataLoader(dataset, batch_size=8)
    for batch in dataloader:
        frames = batch["frames"]
        audio = batch["waveform"]
        rms = batch["rms"]
        prompt = batch["prompt"]
        start_time = batch["start_time"]
        end_time = batch["end_time"]
        print(frames.shape, audio.shape, rms.shape, prompt, start_time, end_time)
        break