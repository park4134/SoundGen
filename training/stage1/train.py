# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import GreatestHitsDataset
from utils.opticalflow import extract_optical_flow_farneback_drumstick
from models.stage1.model import get_model
import os
import datetime
import argparse
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.loss import CrossEntropyLossWithGaussianSmoothedLabels, tolerant_accuracy, FocalLossWithGaussianSmoothedLabels
from preprocess.data_utils import RMS



mu_bins = RMS.get_mu_bins(
    mu=255,
    num_bins=64,
    rms_min=0.01)

def train_one_epoch(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_tol_acc = 0.0
    total_steps = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        rgb = batch["frames"].permute(0, 2, 1, 3, 4).to(device)  # (N, 3, T, H, W)
        flow = batch["flow"].permute(0, 2, 1, 3, 4).to(device)   # (N, 2, T, H, W)
        onset = batch["onset_times"].to(device)
        # target = batch["peak"].to(device)
        target = batch["rms_class"].to(device)  # classification labels


        optimizer.zero_grad()
        output = model(rgb, flow, onset)
        
        
        # N, C, T = output.shape
        # output_flat = output.permute(0, 2, 1).reshape(-1, C)  # (N*T, C)
        # target_flat = target.reshape(-1)  # (N*T,)


        # # mask = target != 0
        # mask_flat = target_flat != 0
        loss = criterion(output, target)
        probs = torch.softmax(output, dim=1)  # (N, C, T)
        preds = torch.argmax(probs, dim=1)    # (N, T)
        print('pred', preds)
        print('GT', target)
        # # loss = criterion(output[mask], target[mask])
        # if mask_flat.sum() == 0:
        #     continue
        # loss = criterion(output_flat[mask_flat], target_flat[mask_flat])
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # standard accuracy (ignore class 0)
            step = epoch * len(dataloader) + batch_idx 
            preds = torch.argmax(output, dim=1)  # (N, T)
            mask = target != 0
            if mask.any():
                acc = (preds[mask] == target[mask]).float().mean().item()
            else:
                acc = 0.0
            # tolerant accuracy over full sequence
            tol_acc = tolerant_accuracy(output, target).item()

            if batch_idx % 10 == 0:
                sample_idx = 0  # 배치 내 첫 샘플만 시각화
                pred_cls = preds[sample_idx]     # (T,)
                gt_cls = target[sample_idx]      # (T,)

                # RMS 값으로 변환
                pred_rms = RMS.undiscretize_rms(pred_cls, mu_bins).cpu().numpy()
                gt_rms = RMS.undiscretize_rms(gt_cls, mu_bins).cpu().numpy()

                # 그래프 이미지 생성 및 로그
                rms_image = plot_rms_comparison(gt_rms, pred_rms, title=f"Epoch {epoch} - Sample RMS")
                writer.add_image("Train/RMS_GT_vs_Pred", rms_image.transpose(2, 0, 1), step)
            
            
            

        #logging
        running_loss += loss.item()
        running_acc += acc
        running_tol_acc += tol_acc
        total_steps += 1
        
        step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/Step_Loss", loss.item(), step)
        writer.add_scalar("Train/Step_Acc", acc, step)
        writer.add_scalar("Train/Step_TolAcc", tol_acc, step)
    
    epoch_loss = running_loss / total_steps
    epoch_acc = running_acc / total_steps
    epoch_tol_acc = running_tol_acc / total_steps

    # epoch_loss = running_loss / len(dataloader)
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Epoch_Acc", epoch_acc, epoch)
    writer.add_scalar("Train/Epoch_TolAcc", epoch_tol_acc, epoch)
    return epoch_loss


def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_tol_acc = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            rgb = batch["frames"].permute(0, 2, 1, 3, 4).to(device)
            flow = batch["flow"].permute(0, 2, 1, 3, 4).to(device)   
            onset = batch["onset_times"].to(device)
            # target = batch["peak"].to(device)
            target = batch["rms_class"].to(device)

            output = model(rgb, flow, onset)
            
            # N, C, T = output.shape
            # output_flat = output.permute(0, 2, 1).reshape(-1, C)  # (N*T, C)
            # target_flat = target.reshape(-1)  # (N*T,)
            
            # mask_flat = target_flat != 0
            # if mask_flat.sum() == 0:
            #     continue
            # loss = criterion(output_flat[mask_flat], target_flat[mask_flat])
            
            # mask = target != 0
            # loss = criterion(output[mask], target[mask])
            
            loss = criterion(output, target)
            # metrics
            preds = torch.argmax(output, dim=1)
            mask = target != 0
            if mask.any():
                acc = (preds[mask] == target[mask]).float().mean().item()
            else:
                acc = 0.0
            tol_acc = tolerant_accuracy(output, target).item()


            # accumulate
            running_loss += loss.item()
            running_acc += acc
            running_tol_acc += tol_acc
            total_steps += 1
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Val/Step_Loss", loss.item(), step)
            writer.add_scalar("Val/Step_Acc", acc, step)
            writer.add_scalar("Val/Step_TolAcc", tol_acc, step)

    epoch_loss = running_loss / total_steps
    epoch_acc = running_acc / total_steps
    epoch_tol_acc = running_tol_acc / total_steps
    writer.add_scalar("Val/Epoch_Loss", epoch_loss, epoch)
    writer.add_scalar("Val/Epoch_Acc", epoch_acc, epoch)
    writer.add_scalar("Val/Epoch_TolAcc", epoch_tol_acc, epoch)
    return epoch_loss



def main(config, mode="classification", date=None):
    

    
    # Load config

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Prepare save directories
    date_str = args.date or datetime.datetime.now().strftime("%Y%m%d")
    run_name = f"{args.mode}_{date_str}"
    run_dir = os.path.join(config["save_dir"], run_name)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard Writer
    # log_dir = os.path.join(config["save_dir"], "logs")
    
    writer = SummaryWriter(log_dir=log_dir)

    # Dataset & DataLoader
    train_dataset = GreatestHitsDataset(
        root_dir=config["root_dir"],
        split_file_path=config["train_split"],
        chunk_length_sec=5.0,
        image_size=(112, 112),
    )
    val_dataset = GreatestHitsDataset(
        root_dir=config["root_dir"],
        split_file_path=config["val_split"],
        chunk_length_sec=5.0,
        image_size=(112, 112),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = get_model(pretrained=config["pretrained"], use_onset=config["use_onset"], use_flow=config["use_flow"]).to(device)
    # criterion = nn.MSELoss()
    # criterion = CrossEntropyLossWithGaussianSmoothedLabels(
    #     num_classes=config.get("n_rms_classes", 64),
    #     blur_range=config.get("gls_blur_range", 8),
    #     sigma=config.get("gls_sigma", 1.0)
    # )
    class_weights = compute_class_weights_from_dataset(train_dataset, config.get("n_rms_classes", 64))

    criterion = FocalLossWithGaussianSmoothedLabels(
    num_classes=config.get("n_rms_classes", 64),
    gamma=2.0,  # 초반에 쉬운 샘플 무시
    blur_range=config.get("gls_blur_range", 8),
    sigma=config.get("gls_sigma", 1.0),
    class_weights=class_weights  # 불균형 대응
    )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])

    os.makedirs(config["save_dir"], exist_ok=True)

    for epoch in range(1, config.get("epochs", 100) + 1):
        print(f"Epoch {epoch}/{config['epochs']} - Run: {run_name}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        val_loss = validate(model, val_loader, criterion, device, writer, epoch)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint in run_dir
        ckpt_path = os.path.join(run_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    writer.close()



def compute_class_weights_from_dataset(dataset, num_classes):
    counts = np.zeros(num_classes)

    for i in range(len(dataset)):
        labels = dataset[i]['rms_class']  # (T,)
        for label in labels:
            if label >= 0 and label < num_classes:
                counts[int(label)] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize to mean=1
    return torch.tensor(weights, dtype=torch.float)

import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def plot_rms_comparison(gt_rms, pred_rms, title="RMS Comparison"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(gt_rms, label='GT', linewidth=2)
    ax.plot(pred_rms, label='Pred', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('RMS')
    ax.legend()
    ax.grid(True)
    
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    plt.close(fig)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="classification", help="Run mode name")
    parser.add_argument("--date", type=str, default=None, help="Date string (YYYYMMDD), default today")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config, mode=args.mode, date=args.date)