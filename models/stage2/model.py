import sys
sys.path.append('./stable_audio_controlnet')  # '-' 포함된 폴더 직접 경로 추가

import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.models.video as video_models

from preprocess.data_utils import RMS
from stable_audio_controlnet.main.controlnet.pretrained import get_pretrained_controlnet_model

def get_controlnet():
    model, model_config = get_pretrained_controlnet_model("stabilityai/stable-audio-open-1.0", controlnet_types=["envelope"], depth_factor=0.2)

    model.model.model.requires_grad_(False)
    model.conditioner.requires_grad_(False)
    model.conditioner.eval()
    model.pretransform.requires_grad_(False)
    model.pretransform.eval()
    
    return model, model_config

class Stage1DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0")
        self._get_image_encoder()

    def _get_image_encoder(self):
        self.r2puls1d = video_models.r2plus1d_18(weights="R2Plus1D_18_Weights.DEFAULT").to(self.device)
        self.r2puls1d.fc = torch.nn.Identity()

        for param in self.r2puls1d.parameters():
            param.requires_grad = False
    
    def forward(self, frames):
        image_feature = self.r2puls1d(frames) # frames shape: (batch, channel, time, height, width)
        return image_feature