import torch
import torch.nn as nn
import numpy as np



import torchvision.models as models

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, large=False):
        super().__init__()
        if large:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
        else:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # 임베딩 벡터 차원 (large: 960, small: 576)
        self.embedding_dim = 960 if large else 576

    def forward(self, x):
        x = self.features(x)          # (N, C, H, W)
        x = self.avgpool(x)           # (N, C, 1, 1)
        x = torch.flatten(x, 1)       # (N, C)
        return x                      # C=embedding_dim

def mobilenetv3_feature_extractor(pretrained=True, large=True):
    """
    MobileNetV3 피쳐 추출 모델 반환
    예시:
        model = mobilenetv3_feature_extractor(pretrained=True, large=True)
        feat = model(x)  # (N, 960) (large), (N, 576) (small)
    """
    return MobileNetV3FeatureExtractor(pretrained=pretrained, large=large)



if __name__ == '__main__':
    model = mobilenetv3_feature_extractor(pretrained=True, large=False)
    x = torch.randn(75, 3, 224, 224)
    features = model(x)
    print(features.shape)  # torch.Size([75, 960]) (large 기준) # torch.Size([75, 576]) (small 기준)