import os
import pandas as pd
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_dir, anno_dir, output_dir, frame_rate=30):
        self.video_dir = video_dir
        self.anno_dir = anno_dir
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.annotations = self.load_annotations()
        self.tool_location_counters  = {} # 각 타격 도구와 위치에 따른 카운터 딕셔너리
    
    def load_annotations(self):
        """Load annotation files into a DataFrame."""
        annotation_files = sorted(
            [os.path.join(self.anno_dir, f) for f in os.listdir(self.anno_dir) if f.endswith('.txt')],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        all_data = []

        for file_path in annotation_files:
            file_number = os.path.basename(file_path).split('.')[0]
            with open(file_path, 'r') as file:
                for line in file:
                    if len(line)==0:
                        continue
                    parts = line.strip().split("\t")

                    hit_time = parts[0]
                    if parts[4] =='':
                        tool_info = parts[3] 
                    else:
                        tool_info = parts[4]
                    tool_parts = tool_info.split('_') #if "_" in tool_info else tool_info.split()
                    hit_location = tool_parts[0].strip()
                    hit_tool = tool_parts[1].strip()
                    all_data.append([hit_time, hit_location, hit_tool, file_number])
        
        df = pd.DataFrame(all_data, columns=["Hit Time", "Hit Location", "Hit Tool", "File Number"])
        return df
    
    def time_to_seconds(self, time_str):
        """Convert time string to seconds."""
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def convert_time_to_frame(self, time_str):
        """Convert time string to frame number using frame rate."""
        seconds = self.time_to_seconds(time_str)
        return int(seconds * self.frame_rate)
    
    def get_tool_location_index(self, hit_tool, hit_location):
        """Get the current index for a hit tool and location combination and increment it."""
        key = (hit_tool, hit_location)
        if key not in self.tool_location_counters:
            self.tool_location_counters[key] = 1
        else:
            self.tool_location_counters[key] += 1
        return self.tool_location_counters[key]
    
    def save_frames_before_hit(self, video_file, hit_time, file_number, hit_tool, hit_location):
        """Save frames before the hit time."""
        video_path = os.path.join(self.video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        hit_frame = self.convert_time_to_frame(hit_time)
        start_frame = max(0, hit_frame - self.frame_rate // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


        # 도구별 하위 폴더 생성
        output_subdir = os.path.join(self.output_dir, hit_tool)
        os.makedirs(output_subdir, exist_ok=True)
        frames = []

        for i in range(self.frame_rate // 2):
            ret, frame = cap.read() # ret : True or False
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if frames:
            frames_array = np.array(frames)
            # 각 hit_tool과 hit_location마다 데이터 인덱스 증가
            data_index = self.get_tool_location_index(hit_tool, hit_location)

            npy_filename = f"{hit_location}_{data_index}_{file_number}_{start_frame}.npy" # 타격위치_데이터인덱스_원본비디오Num_시작프레임
            npy_filepath = os.path.join(output_subdir, npy_filename)

            np.save(npy_filepath, frames_array)
            print(f"Saved {npy_filename} with {len(frames)} frames.")

    def process_videos(self):
        """Process all videos based on annotations."""
        for index, row in self.annotations.iterrows():
            video_file = f"VR_{row['File Number']}.mp4"
            self.save_frames_before_hit(video_file, row["Hit Time"], row["File Number"], row["Hit Tool"], row["Hit Location"])


# 데이터 경로 설정
data_dir = '/home/sangjun/Desktop/Sound_1st/data'
Anno_dir = os.path.join(data_dir, 'Annotation')
Video_dir = os.path.join(data_dir, 'VR')
WAV_dir = os.path.join(data_dir, 'WAV')
Img_dir = os.path.join(data_dir, 'img')
# print('Anno_dir',Anno_dir)

# VideoProcessor 객체 생성 및 비디오 처리 실행
processor = VideoProcessor(Video_dir, Anno_dir, Img_dir)
processor.process_videos()
