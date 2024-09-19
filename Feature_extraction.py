import torch
import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm

# 사용자 정의 데이터셋 클래스 정의
class NpyDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.npy_paths = glob(os.path.join(data_root, '*.npy'), recursive=True) #'**',
        self.npy_paths = sorted(self.npy_paths)  # 정렬
        
    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        data = {}
        npy_path = self.npy_paths[idx]
        images = np.load(npy_path)  # npy 파일에서 이미지 데이터 로드
        # print('images_shape :',images.shape)
        # images = [Image.fromarray(img) for img in images]  # PIL 이미지로 변환

        # 이미지를 PIL Image로 변환 후 transform 적용
        if self.transform:
            images = [self.transform(Image.fromarray(img)) for img in images]  # 각 이미지에 transform 적용
        images = torch.stack(images)  # 리스트를 텐서로 변환 (15, 3, 224, 224)
        data['images'] = images  # shape: (15, 3, 224, 224)
        data['npy_path'] = npy_path
        return data


# 데이터 변환 설정
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 사전 훈련된 모델과 특징 추출기 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
model.eval()

# 데이터셋 및 데이터로더 설정
tool_name = 'rubber'

data_dir = os.path.join('/home/sangjun/Desktop/Sound_1st/data/img/', tool_name)
dataset = NpyDataset(data_root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# 특징을 저장할 루트 디렉토리 설정
feature_root_dir = '/home/sangjun/Desktop/Sound_1st/data/img_feature'
 
tool_feature_dir = os.path.join(feature_root_dir, tool_name)
os.makedirs(tool_feature_dir, exist_ok=True)  # 도구 이름의 폴더 생성


# 데이터 순회하며 특징 추출 및 저장
with torch.no_grad():
    for data in tqdm(dataloader):
        images = data['images'][0]  # 데이터셋에서 (15, 3, 224, 224) 형식으로 변환된 이미지 목록
        print('images__shape:',images.shape)
        npy_path = data['npy_path'][0]
        
        # 이미지를 배치로 모델에 입력하기 위해 리스트 형태로 전달
        inputs = image_processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features = outputs.pooler_output.cpu().numpy()  # (배치 크기, 768) 형태의 이미지 특징 추출
        
        # 기존 npy 파일 이름을 기반으로 저장할 파일 이름 설정
        npy_filename = os.path.basename(npy_path)  # 기존 npy 파일 이름
        feature_filename = npy_filename.replace('.npy', '_features.npy')  # 예: '1_dashboard_1_0.npy' -> '1_dashboard_1_0_features.npy'
        
        # 특징을 저장할 경로 설정
        feature_save_path = os.path.join(tool_feature_dir, feature_filename)
        
        # .npy 파일로 저장 (전체 배치의 특징)
        np.save(feature_save_path, features)
        print(f"Saved features to {feature_save_path}")