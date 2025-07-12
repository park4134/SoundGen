# import torch
# import torch.nn as nn
# import numpy as np



# import torchvision.models as models

# class MobileNetV3FeatureExtractor(nn.Module):
#     def __init__(self, pretrained=True, large=False):
#         super().__init__()
#         if large:
#             weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
#             backbone = models.mobilenet_v3_large(weights=weights)
#         else:
#             weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
#             backbone = models.mobilenet_v3_small(weights=weights)
#         self.features = backbone.features
#         self.avgpool = backbone.avgpool
#         # 임베딩 벡터 차원 (large: 960, small: 576)
#         self.embedding_dim = 960 if large else 576

#     def forward(self, x):
#         x = self.features(x)          # (N, C, H, W)
#         x = self.avgpool(x)           # (N, C, 1, 1)
#         x = torch.flatten(x, 1)       # (N, C)
#         return x                      # C=embedding_dim

# def mobilenetv3_feature_extractor(pretrained=True, large=True):
#     """
#     MobileNetV3 피쳐 추출 모델 반환
#     예시:
#         model = mobilenetv3_feature_extractor(pretrained=True, large=True)
#         feat = model(x)  # (N, 960) (large), (N, 576) (small)
#     """
#     return MobileNetV3FeatureExtractor(pretrained=pretrained, large=large)



# if __name__ == '__main__':
#     model = mobilenetv3_feature_extractor(pretrained=True, large=False)
#     x = torch.randn(75, 3, 224, 224)
#     features = model(x)
#     print(features.shape)  # torch.Size([75, 960]) (large 기준) # torch.Size([75, 576]) (small 기준)

import torch
import torch.nn as nn
from models.stage1.resnet import r2plus1d_18
from utils.opticalflow import preprocess_optical_flow

class R2plus1d18KeepTemp(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        self.model.layer2[0].conv1[0][3] = nn.Conv3d(230, 128, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(128)
        )
        self.model.layer3[0].conv1[0][3] = nn.Conv3d(460, 256, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(256)
        )
        self.model.layer4[0].conv1[0][3] = nn.Conv3d(921, 512, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(512)
        )
        self.model.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        
        return self.model(x)
    
class PeakHead(nn.Module):
    def __init__(self, in_dim: int, n_peak_classes: int, n_onsets: int = 10):
    
        super().__init__()
        self.n_peak_classes = n_peak_classes
        self.n_onsets = n_onsets
        self.net  = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1875*n_peak_classes),
            # nn.ReLU(True)
        )

    # def forward(self, x):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out.view(-1, 1875, self.n_peak_classes)
        return out.transpose(1, 2)

###

####################
# PeakNet
####################
class PeakNet(nn.Module):
    def __init__(self, pretrained=True, use_onset=True, use_flow=True, freeze_backbone=True, n_peak_classes=64, n_onsets=10):
        super().__init__()
        self.rgb_net = R2plus1d18KeepTemp(pretrained=pretrained)
        self.use_flow = use_flow
        self.use_onset = use_onset
        self.n_peak_classes = n_peak_classes
        self.n_onsets = n_onsets


        if freeze_backbone:
            for param in self.rgb_net.parameters():
                param.requires_grad = False

        if use_flow:
            self.flow_net = R2plus1d18KeepTemp(pretrained=pretrained)
            if freeze_backbone:
                for param in self.flow_net.parameters():
                    param.requires_grad = False

        in_dim = 75 * (2 if use_flow else 1)

        if use_onset:
            in_dim += n_onsets
        print('in_dim',in_dim)

        self.head = PeakHead(in_dim=in_dim, n_peak_classes=n_peak_classes, n_onsets=n_onsets)


    def forward(self, rgb, flow, onset_times=None):


        if self.use_flow:
            flow_proc = preprocess_optical_flow(flow)  
            flow_feat = self.flow_net(flow_proc)  
            flow_feat = flow_feat.mean(dim=1).squeeze(-1).squeeze(-1) # [N, 75]
            
            rgb_feat = self.rgb_net(rgb)        
            rgb_feat = rgb_feat.mean(dim=1).squeeze(-1).squeeze(-1)   # [N, 75]
            
            feats = [rgb_feat, flow_feat]
        else:
            rgb_feat = self.rgb_net(rgb)        
            rgb_feat = rgb_feat.mean(dim=1).squeeze(-1).squeeze(-1)   # [N, 75]
            
            feats = rgb_feat
        
    
        # print("rgb_feat:", rgb_feat.shape)
        # print("flow_feat:", flow_feat.shape)
        
        # print("rgb_feat_mean_squeeze:", rgb_feat.shape)
        # print("flow_feat_mean_squeeze:", flow_feat.shape)


        if self.use_onset:
            # assert onset_times is not None, "onset_times 인풋 필요"
            feats.append(onset_times)  # [N, 10]
        x = torch.cat(feats, dim=-1) # [N, 1024 or 1034]
        out = self.head(x)           # [N, 10]

        return out


###################
# 테스트 예시
###################
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, T, H, W = 8, 75, 112, 112
    rgb = torch.randn(N, 3, T, H, W).to(device)
    flow = torch.randn(N, 2, T, H, W).to(device)
    onset_times = torch.rand(N, 10).to(device)
    model = PeakNet(pretrained=False, use_onset=True).to(device)
    model.eval()
    with torch.no_grad():
        out = model(rgb, flow, onset_times)
    print("out:",out)
    print("peak output:", out.shape)  # (N, 10)



