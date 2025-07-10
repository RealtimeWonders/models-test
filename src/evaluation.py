import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)

def calculate_psnr(video1, video2):
    if video1.shape != video2.shape:
        print(f"Warning: Video shapes don't match: {video1.shape} vs {video2.shape}")
        min_frames = min(video1.shape[0], video2.shape[0])
        video1 = video1[:min_frames]
        video2 = video2[:min_frames]
    
    psnr_values = []
    for i in range(len(video1)):
        psnr_val = psnr(video1[i], video2[i])
        psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)

def calculate_ssim(video1, video2):
    if video1.shape != video2.shape:
        print(f"Warning: Video shapes don't match: {video1.shape} vs {video2.shape}")
        min_frames = min(video1.shape[0], video2.shape[0])
        video1 = video1[:min_frames]
        video2 = video2[:min_frames]
    
    ssim_values = []
    for i in range(len(video1)):
        ssim_val = ssim(video1[i], video2[i], multichannel=True, channel_axis=-1)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def main():
    parser = argparse.ArgumentParser(description='Evaluate video quality metrics')
    parser.add_argument('--video', required=True, help='Path to generated video')
    parser.add_argument('--reference', required=True, help='Path to reference video')
    args = parser.parse_args()
    
    print(f"Loading videos...")
    video = load_video(args.video)
    reference = load_video(args.reference)
    
    print(f"Video shape: {video.shape}")
    print(f"Reference shape: {reference.shape}")
    
    psnr_score = calculate_psnr(video, reference)
    ssim_score = calculate_ssim(video, reference)
    
    print(f"\nEvaluation Results:")
    print(f"PSNR: {psnr_score:.2f} dB")
    print(f"SSIM: {ssim_score:.4f}")

if __name__ == '__main__':
    main()