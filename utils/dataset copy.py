import torch
import torchvision.transforms as transforms
from utils.opticalflow import extract_optical_flow_farneback_drumstick
from utils.opticalflow import preprocess_optical_flow
import os
from glob import glob
from natsort import natsorted
from PIL import Image
import numpy as np
import json
import torchaudio
import librosa


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
        peak_file_suffix="rms.npy",
        frame_file_suffix=".png",
        transform_=None,
        use_peak=True,        
        n_peak_classes=64,
        mu_peak=255,
        rms_num_bins = 64,
        rms_min = 0.01
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
        self.n_peak_classes = n_peak_classes
        self.mu_peak = mu_peak
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
                peak_path = os.path.join(chunk_dir, self.peak_file_suffix)
                audio_path = os.path.join(chunk_dir, self.audio_file_suffix)
                
                self.samples.append({
                    "video_name": video_name,
                    "chunk_dir": chunk_dir,
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
        peak_array = np.zeros(10, dtype=np.float32)     
        
        onset = None
        if os.path.exists(sample["onset_path"]):
            onset = np.load(sample["onset_path"])

        peak = None
        if self.use_peak and os.path.exists(sample["peak_path"]):
            peak = np.load(sample["peak_path"])
            
            
        if onset is not None and peak is not None:
            onset_indices = np.where(onset > 0)[0]
            num_onsets = min(10, len(onset_indices))
            
            for i in range(num_onsets):
                idx = onset_indices[i]
                time_sec = idx / 240000
                onset_time_array[i] = time_sec
                peak_array[i] = peak[idx]  # same index


        # peak 분류 레이블 생성
        if self.use_peak:
            peak_classes = 




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
            "peak": torch.tensor(peak_array), #
            'peak_class': torch.tensor(peak_classes, dtype=torch.long),
            "waveform": wav,
            "audio_sr": sr,
        }
        

if __name__ == "__main__":
    dataset = GreatestHitsDataset(
        root_dir="/mnt/HDD2/GreatestHits/preprocessed_15_5.0_(320, 240)_48000",
        split_file_path="/mnt/HDD2/GreatestHits/data/train.txt",
        chunk_length_sec=5.0,
        image_size=(112, 112),
    )
    
    print("총 chunk 샘플:", len(dataset))
    item = dataset[100]
    print("frames:", item["frames"].shape)
    print("onset_times:", item["onset_times"])
    print("text:", item["text"])
    print("start/end:", item["start_time"], item["end_time"])
    print("peak:", item["peak"])
    print("peakclass:", item["peak_class"])
    print("waveform:", item["waveform"].shape if item["waveform"] is not None else None)
                

    import matplotlib.pyplot as plt

#     all_peak = []

#     for i in range(len(dataset)):
#         item = dataset[i]
#         peak = item["peak"]
#         peak = np.asarray(peak)
#         all_peak.append(peak)

#     all_peak = np.concatenate(all_peak)
#     print(f"peak 값 shape: {all_peak.shape}")
#     print(f"0 값 개수: {(all_peak==0).sum()}, 전체 대비 {(all_peak==0).mean()*100:.2f}%")

#     # 원본 peak 분포
#     plt.figure()
#     plt.hist(all_peak, bins=100, log=True)
#     plt.title("Peak Value Distribution (original)")
#     plt.xlabel("peak")
#     plt.ylabel("Count (log scale)")
#     plt.grid()
#     plt.show()

#     # µ-law 인코딩 후 분포 (Librosa)
#     import librosa
#     mu = 255
#     peak_mu = librosa.mu_compress(all_peak, mu=mu)
#     plt.figure()
#     plt.hist(peak_mu, bins=100, log=True)
#     plt.title("Peak Value Distribution (after µ-law, mu=255)")
#     plt.xlabel("mu-law encoded peak")
#     plt.ylabel("Count (log scale)")
#     plt.grid()
# plt.show()