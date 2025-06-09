import torch
import torchvision.transforms as transforms
import os
from glob import glob
from natsort import natsorted
from PIL import Image
import numpy as np
import json
import torchaudio


class GreatestHitsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split_file_path,
        chunk_length_sec=5.0,
        image_size=(224,224),
        audio_file_suffix=".resampled.wav",
        onset_file_suffix=".onset.npy",
        metadata_file_suffix=".metadata.json",
        peak_file_suffix=".rms.npy",
        frame_file_suffix=".png",
        transform_=None,
        use_peak=True,        
    ):
        super().__init__()
        self.root_dif = root_dir
        self.chunk_length_sec = chunk_length_sec
        self.audio_file_suffix = audio_file_suffix
        self.onset_file_suffix = onset_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.peak_file_suffix = peak_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.use_peak = use_peak
        
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
            video_base_dir = os.path.join(root_dir, video_name)
            chunk_dirs = natsorted(glob(os.path.join(video_base_dir, "chunk_*")))

            for chunk_dir in chunk_dirs:
                chunk_subdir = os.path.join(chunk_dir, video_name)
                if not os.path.isdir(chunk_subdir):
                    continue
                frames_dir = os.path.join(chunk_subdir, "frames")
                frames = natsorted(glob(os.path.join(frames_dir, f"*{self.frame_file_suffix}")))
                if not frames:
                    continue
                    
                onset_path = os.path.join(chunk_subdir, video_name + self.onset_file_suffix)
                meta_path = os.path.join(chunk_subdir, video_name + self.metadata_file_suffix)
                peak_path = os.path.join(chunk_subdir, video_name + self.peak_file_suffix)
                audio_path = os.path.join(chunk_subdir, video_name + self.audio_file_suffix)
                
                self.samples.append({
                    "video_name": video_name,
                    "chunk_dir": chunk_subdir,
                    "frames": frames,
                    "onset_path": onset_path,
                    "meta_path": meta_path,
                    "peak_path": peak_path,
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
        imgs = torch.stack(imgs, dim=0) #(T, C, H, W)
        
        onset = None
        if os.path.exists(sample["onset_path"]):
            onset = np.load(sample["onset_path"])

        peak = None
        if self.use_peak and os.path.exists(sample["peak_path"]):
            peak = np.load(sample["peak_path"])

        text = ""
        start_time = 0.0
        end_time = 0.0
        if os.path.exists(sample["meta_path"]):
            with open(sample["meta_path"], "r") as f:
                meta = json.load(f)
            text = meta.get("event_text", "")
            start_time = meta.get("chunk_start", 0.0)
            end_time = meta.get("chunk_end", 0.0)

        wav = None
        if os.path.exists(sample["audio_path"]):
            wav, sr = torchaudio.load(sample["audio_path"])
        else:
            sr = None

        return {
            "video_name": sample["video_name"],
            "frames": imgs,
            "onset_times": torch.tensor(onset) if onset is not None else None,
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "peak": torch.tensor(peak) if peak is not None else None,
            "waveform": wav,
            "audio_sr": sr,
        }

if __name__ == "__main__":
    dataset = GreatestHitsDataset(
        root_dir="/mnt/HDD2/GreatestHits/preprocessed_15_5.0_(320,240)_48000",
        split_file_path="/mnt/HDD2/GreatestHits/data/train.txt",
        chunk_length_sec=5.0,
        image_size=(320, 240),
    )
    
    print("총 chunk 샘플:", len(dataset))
    item = dataset[0]
    print("frames:", item["frames"].shape)
    print("onset_times:", item["onset_times"])
    print("text:", item["text"])
    print("start/end:", item["start_time"], item["end_time"])
    print("peak:", item["peak"])
    print("waveform:", item["waveform"].shape if item["waveform"] is not None else None)
                
                