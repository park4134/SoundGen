# model.py
import torch
import torch.nn as nn
from models.stage1.module import PeakNet


class PeakRegressionModel(nn.Module):
    def __init__(self, pretrained=True, use_onset=True, use_flow=True):
        super().__init__()
        self.model = PeakNet(pretrained=pretrained, use_onset=use_onset)

    def forward(self, rgb, flow, onset_times):
        """
        Args:
            rgb: [N, 3, T, H, W]
            flow: [N, 2 or 3, T, H, W]
            onset_times: [N, 10]
        Returns:
            peak_pred: [N, 10]
        """
        return self.model(rgb, flow, onset_times)

def get_model(pretrained=True, use_onset=True, use_flow=True):
    return PeakRegressionModel(pretrained=pretrained, use_onset=use_onset)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, T, H, W = 4, 75, 112, 112
    rgb = torch.randn(N, 3, T, H, W).to(device)
    flow = torch.randn(N, 2, T, H, W).to(device)
    onset_times = torch.rand(N, 10).to(device)
    model = get_model(pretrained=False, use_onset=True).to(device)
    model.eval()
    # with torch.no_grad():
    out = model(rgb, flow, onset_times)
    print("peak output:", out.shape)
