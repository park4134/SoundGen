import cv2
import numpy as np
import torch
from dataset import GreatestHitsDataset
import os

#############################
# 드럼스틱 마스크 함수 (HSV/BGR 둘다 지원)
#############################
def get_drumstick_mask(img_bgr, mode="hsv", lower=None, upper=None):
    """
    img_bgr: (H, W, 3) BGR uint8
    mode: "hsv" or "bgr"
    lower, upper: np.array([..]) 형태, 마스킹 범위
    """
    if mode == "hsv":
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower, upper)
    elif mode == "bgr":
        mask = cv2.inRange(img_bgr, lower, upper)
    else:
        raise ValueError("mode은 'hsv' 또는 'bgr'만 지원")
    return mask  # 0 or 255 (white area: 드럼스틱)

#############################
# optical flow + 마스크 적용
#############################
def extract_optical_flow_farneback_drumstick(frames, mode="hsv", lower=None, upper=None):
    """
    frames: (T, C, H, W) torch.Tensor
    mode: "hsv" or "bgr"
    lower, upper: 드럼스틱 색상 범위
    return: flow_list, hsv_list, mask_list
    """
    if isinstance(frames, torch.Tensor):
        imgs_rgb = (frames.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
    else:
        imgs_rgb = frames  # (T, H, W, 3)
    if len(imgs_rgb) < 2:
        print("프레임이 2개 이상 필요합니다.")
        return [], [], []
    flow_list, hsv_list, mask_list = [], [], []
    prev_bgr = cv2.cvtColor(imgs_rgb[0], cv2.COLOR_RGB2BGR)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    prev_mask = get_drumstick_mask(prev_bgr, mode, lower, upper)
    for i in range(1, len(imgs_rgb)):
        curr_bgr = cv2.cvtColor(imgs_rgb[i], cv2.COLOR_RGB2BGR)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        curr_mask = get_drumstick_mask(curr_bgr, mode, lower, upper)
        mask = cv2.bitwise_and(prev_mask, curr_mask)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.3, levels=3, winsize=3,
            iterations=3, poly_n=7, poly_sigma=1.5,
            flags=0
        )
        # 드럼스틱 영역만 flow 남기기
        mask_bool = mask > 0
        flow_masked = np.zeros_like(flow)
        flow_masked[mask_bool] = flow[mask_bool]
        # HSV 시각화
        mag, ang = cv2.cartToPolar(flow_masked[..., 0], flow_masked[..., 1])
        hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_list.append(flow_masked)
        hsv_list.append(hsv)
        mask_list.append(mask)
        prev_gray, prev_mask = curr_gray, curr_mask
    return flow_list, hsv_list, mask_list

#########################
# Overlay, 영상 저장 함수
#########################
def overlay_flow_on_image(rgb_img, flow_hsv, alpha=0.6):
    if flow_hsv.shape[2] == 3 and np.max(flow_hsv[...,0]) <= 180:
        flow_rgb = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)
    else:
        flow_rgb = flow_hsv
    flow_rgb = cv2.resize(flow_rgb, (rgb_img.shape[1], rgb_img.shape[0]))
    blended = cv2.addWeighted(rgb_img, alpha, flow_rgb, 1-alpha, 0)
    return blended

def save_flow_overlay_video(rgb_frames, hsvs, out_path="flow_overlay.mp4", alpha=0.6, fps=15):
    if isinstance(rgb_frames, torch.Tensor):
        rgb_frames = (rgb_frames.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = rgb_frames[0].shape[:2]
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for i in range(1, len(rgb_frames)):
        img = rgb_frames[i]
        flow_hsv = hsvs[i-1]
        overlay = overlay_flow_on_image(img, flow_hsv, alpha=alpha)
        out.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"영상 저장 완료: {out_path}")

#####################################
# MAIN (예시)
#####################################
if __name__ == '__main__':
    # 데이터셋 준비
    dataset = GreatestHitsDataset(
        root_dir="/mnt/HDD2/GreatestHits/preprocessed_15_5.0_(320,240)_48000",
        split_file_path="/mnt/HDD2/GreatestHits/data/train.txt",
        chunk_length_sec=5.0,
        image_size=(320, 240),
    )
    item = dataset[1]
    frames = item["frames"]  # (T, C, H, W) torch.Tensor

    # ====== HSV 방식 ======
    # hsv_lower = np.array([10, 90, 70])
    # hsv_upper = np.array([27, 200, 180])
    # flows, hsvs, masks = extract_optical_flow_farneback_drumstick(frames, mode="hsv", lower=hsv_lower, upper=hsv_upper)

    # ====== BGR 방식 (예시) ======
    bgr_lower = np.array([20, 20, 0]) 
    bgr_upper = np.array([255, 255, 190])
    flows, hsvs, masks = extract_optical_flow_farneback_drumstick(frames, mode="bgr", lower=bgr_lower, upper=bgr_upper)

    # 영상 프레임 torch → numpy (T,H,W,3, uint8)로 변환
    rgb_frames = (frames.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
    os.makedirs("test_flow_vis", exist_ok=True)
    save_flow_overlay_video(rgb_frames, hsvs, out_path="test_flow_vis/flow_overlay_stick.mp4", alpha=0.6, fps=15)

    # (옵션) 마스크/flow 시각화 저장
    for i in range(min(5, len(masks))):
        cv2.imwrite(f"test_flow_vis/drumstick_mask_{i:02d}.png", masks[i])
        cv2.imwrite(f"test_flow_vis/drumstick_flow_hsv_{i:02d}.png", hsvs[i])

    print("드럼스틱 기반 optical flow 영상 및 마스크 시각화 완료")
