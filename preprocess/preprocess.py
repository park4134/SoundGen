import os
import json
import librosa
import argparse
import subprocess
import numpy as np
import pandas as pd
import soundfile as sf

from glob import glob
from tqdm import tqdm
from natsort import natsorted

class Preprocessor():
    def __init__(self):
        self.parse_arg()
        self.get_paths()

    def parse_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', type=str, default='greatest_hits')
        parser.add_argument('--video_sample_rate', type=int, default=15)
        parser.add_argument('--audio_sample_rate', type=int, default=48000)
        parser.add_argument('--clip_duration', type=float, default=5.0)
        # parser.add_argument('--offset_from_event', type=float, default=0.5)
        parser.add_argument('--image_size', nargs='+', type=int, default=[320, 240])
        parser.add_argument('--rms_window_size', type=int, default=512)
        parser.add_argument('--rms_scale_factor', type=float, default=2.0)
        parser.add_argument('--prompt_mode', type=str, default='list')
        self.args = parser.parse_args()

    def get_paths(self):
        base_path = os.path.join(os.getcwd(), 'data', self.args.data_dir)
        self.video_paths = natsorted(glob(os.path.join(base_path, 'high', '*_mic.mp4')))
        self.audio_paths = natsorted(glob(os.path.join(base_path, 'high', '*_denoised.wav')))
        self.anno_paths = natsorted(glob(os.path.join(base_path, 'high', '*_times.txt')))
        self.output_base_path = os.path.join(base_path, f'preprocessed_{self.args.video_sample_rate}_{self.args.clip_duration}_{tuple(self.args.image_size)}_{self.args.audio_sample_rate}')
        os.makedirs(self.output_base_path, exist_ok=True)

    def get_annotation(self, path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()

                event_time = float(parts[0])
                material = parts[1] if parts[1] != 'None' else None
                event = parts[2] if parts[2] != 'None' else None
                motion = parts[3] if parts[3] != 'None' else None
                
                data.append({
                    'event_time': event_time,
                    'material': material,
                    'event': event,
                    'motion': motion
                })

        return pd.DataFrame(data)
    
    def probe_video(self, video_path):
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'stream',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def get_videoNaudio_info(self, video_path):
        metadata_in = self.probe_video(video_path)
        metadata_out = {
        "original": {
            "width": int(metadata_in["streams"][0]["width"]),
            "height": int(metadata_in["streams"][0]["height"]),
            "video_frame_rate": float(eval(metadata_in["streams"][0]["avg_frame_rate"])),
            "video_duration": float(metadata_in["streams"][0]["duration"]),
            "video_num_frames": int(metadata_in["streams"][0]["nb_frames"]),
            "audio_sample_rate": int(metadata_in["streams"][1]["sample_rate"]),
            "audio_channels": int(metadata_in["streams"][1]["channels"]),
            "audio_duration": float(metadata_in["streams"][1]["duration"]),
            "audio_num_samples": int(metadata_in["streams"][1]["duration_ts"]),
            },
        "processed": {
            "width": int(self.args.image_size[0]),
            "height": int(self.args.image_size[1]),
            "video_frame_rate": int(self.args.video_sample_rate),
            "video_duration": float(metadata_in["streams"][0]["duration"]),
            "video_num_frames": int(float(metadata_in["streams"][0]["duration"]) * self.args.video_sample_rate),
            "audio_sample_rate": int(self.args.audio_sample_rate),
            "audio_channels": 1,
            "audio_bitdepth": 32,
            }
        }
        return metadata_out

    def resample_audio(self, wav_path, audio_sample_rate):
        cmd = [
            'ffmpeg',
            '-i', wav_path,
            '-f', 'f32le',                 # raw 32-bit float PCM
            '-acodec', 'pcm_f32le',        # explicitly set codec
            '-ar', str(audio_sample_rate), # target sample rate
            '-ac', '1',                    # mono
            '-'
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_audio, err = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{err.decode()}")

        # Convert raw bytes to NumPy array (float32)
        audio_data = np.frombuffer(raw_audio, np.float32)

        return audio_data
    
    def get_rmsNonset(self, chunk_start_time, chunk_end_time):
        self.onset_array = np.zeros_like(self.audio_chunk, dtype=np.float32)

        rms_values = librosa.feature.rms(y=self.audio_chunk, frame_length=self.args.rms_window_size, hop_length=self.args.rms_window_size).flatten()
        rms_full = np.repeat(rms_values, self.args.rms_window_size)
        rms_full = rms_full[:len(self.audio_chunk)]

        # 이벤트 처리
        self.anno_in_chunk = self.df_anno[
            (self.df_anno['event_time'] >= chunk_start_time) &
            (self.df_anno['event_time'] < chunk_end_time)
            ]
        
        self.rms_array = np.zeros_like(self.audio_chunk, dtype=np.float32)
        margin_samples = 100

        if len(self.anno_in_chunk) > 0:
            for event_time in self.anno_in_chunk['event_time']:
                event_sample_idx = int((event_time - chunk_start_time) * self.args.audio_sample_rate)
                start_idx = max(0, event_sample_idx - margin_samples)
                end_idx = min(len(self.audio_chunk), event_sample_idx + margin_samples + 1)

                self.onset_array[event_sample_idx] = 1.0

                if start_idx < end_idx:
                    margin_rms_chunk = rms_full[start_idx:end_idx]
                    local_max_idx = np.argmax(margin_rms_chunk)
                    max_pos = start_idx + local_max_idx
                    self.rms_array[max_pos] = rms_full[max_pos]
    
    def get_frames(self, frame_dir, chunk_start_time, path_v):
        frame_output_pattern = os.path.join(frame_dir, 'frame_%05d.png')
        ffmpeg_cmd = (
            f'ffmpeg -ss {chunk_start_time} -t {self.args.clip_duration} '
            f'-i \"{path_v}\" '
            f'-vf \"fps={self.args.video_sample_rate},scale={self.args.image_size[0]}:{self.args.image_size[1]}\" '
            f'\"{frame_output_pattern}\" -hide_banner -loglevel error'
        )
        os.system(ffmpeg_cmd)

    def get_prompt(self, frequency="multiple times"):
        df_clean = self.df_anno[['event', 'material']].dropna().drop_duplicates()

        if len(df_clean) == 0:
            return ""

        unique_events = df_clean['event'].unique().tolist()
        unique_materials = df_clean['material'].unique().tolist()

        if len(unique_events) == 1 and len(unique_materials) == 1:
            return f"A person {unique_events[0]} {frequency} on {unique_materials[0]} with a wooden stick."

        if self.args.prompt_mode == "list":
            event_material_map = (
                df_clean.groupby("event")["material"]
                .apply(list)
                .to_dict()
            )

            event_phrases = []
            for event, materials in event_material_map.items():
                if len(materials) == 1:
                    material_str = materials[0]
                    event_phrases.append(f"{event}s {material_str}")
                elif len(materials) == 2:
                    material_str = f"{materials[0]} and {materials[1]}"
                    event_phrases.append(f"{event}s {material_str}")
                elif len(materials) >= 3:
                    material_str = ", ".join(materials[:-1]) + f", and {materials[-1]}"
                    event_phrases.append(f"{event}s {material_str}")

            if len(event_phrases) == 1:
                combined = event_phrases[0]
            elif len(event_phrases) == 2:
                combined = f"{event_phrases[0]} and {event_phrases[1]}"
            else:
                combined = ", ".join(event_phrases[:-1]) + f", and {event_phrases[-1]}"

            return f"A person {combined} {frequency} with a wooden stick."

        else:
            raise ValueError("prompt_mode must be either 'list' or 'all_comb'.")

    def process_chunks(self, output_dir, path_v):
        audio_chunk_size = int(self.args.audio_sample_rate * self.args.clip_duration)

        num_video_chunks = int(self.info['processed']['video_duration'] // self.args.clip_duration)
        num_audio_chunks = len(self.resampled_audio) // audio_chunk_size

        for i in range(min(num_video_chunks, num_audio_chunks)):
            # 청크 구간 계산
            chunk_start_time = i * self.args.clip_duration
            chunk_end_time = (i + 1) * self.args.clip_duration

            # 오디오 청크
            self.audio_chunk = self.resampled_audio[i * audio_chunk_size : (i + 1) * audio_chunk_size]

            self.get_rmsNonset(chunk_start_time, chunk_end_time)
            
            # 출력 경로 준비
            chunk_dir = os.path.join(output_dir, f'chunk_{i}', path_v.split(os.sep)[-1].split('_')[0])
            frame_dir = os.path.join(chunk_dir, 'frames')
            os.makedirs(frame_dir, exist_ok=True)

            self.get_frames(frame_dir, chunk_start_time, path_v) # Video frame 저장
            
            prompt = self.get_prompt(frequency="multiple times")
            metadata = {
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": 5
                }
            with open(os.path.join(chunk_dir, 'metadata.json'), "w") as f:
                json.dump(metadata, f, indent=4)

            np.save(os.path.join(chunk_dir, 'peak.npy'), self.rms_array) # RMS 값 저장
            sf.write(os.path.join(chunk_dir, 'resampled.wav'), self.audio_chunk, self.args.audio_sample_rate) # Resampled audio 저장
            np.save(os.path.join(chunk_dir, 'onset.npy'), self.onset_array) # Onset 저장

    def preprocess(self):
        for path_v, path_a, path_anno in tqdm(zip(self.video_paths, self.audio_paths, self.anno_paths), total=len(self.video_paths)):
            # 비디오, 오디오 정보 추출
            self.info = self.get_videoNaudio_info(path_v)
            self.df_anno = self.get_annotation(path_anno)
            
            # 오디오 리샘플링
            self.resampled_audio = self.resample_audio(wav_path=path_a, audio_sample_rate=self.args.audio_sample_rate)

            # 청크 나누기 및 저장
            output_dir = os.path.join(self.output_base_path)
            self.process_chunks(output_dir, path_v)

if __name__=="__main__":
    PP = Preprocessor()
    PP.preprocess()