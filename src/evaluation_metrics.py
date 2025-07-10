import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from PIL import Image

class VideoEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_video_metrics(self, video_path: str, reference_image_path: str = None) -> Dict:
        """Calculate various metrics for video quality assessment"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return {"error": "No frames found in video"}
        
        metrics = {
            "total_frames": len(frames),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "duration": len(frames) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
            "temporal_consistency": self._calculate_temporal_consistency(frames),
            "motion_smoothness": self._calculate_motion_smoothness(frames),
            "visual_quality": self._calculate_visual_quality(frames),
        }
        
        if reference_image_path:
            metrics["fidelity_to_input"] = self._calculate_fidelity_to_input(frames[0], reference_image_path)
        
        return metrics
    
    def _calculate_temporal_consistency(self, frames: List[np.ndarray]) -> float:
        """Calculate temporal consistency between consecutive frames"""
        if len(frames) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(frames) - 1):
            frame1_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM between consecutive frames
            similarity = ssim(frame1_gray, frame2_gray)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_motion_smoothness(self, frames: List[np.ndarray]) -> float:
        """Calculate motion smoothness using optical flow"""
        if len(frames) < 2:
            return 0.0
        
        flow_magnitudes = []
        for i in range(len(frames) - 1):
            frame1_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                frame1_gray, frame2_gray, 
                cv2.goodFeaturesToTrack(frame1_gray, maxCorners=100, qualityLevel=0.01, minDistance=10),
                None
            )[0]
            
            if flow is not None:
                flow_magnitude = np.mean(np.linalg.norm(flow, axis=2))
                flow_magnitudes.append(flow_magnitude)
        
        # Lower variance in flow magnitudes indicates smoother motion
        return 1.0 / (1.0 + np.var(flow_magnitudes)) if flow_magnitudes else 0.0
    
    def _calculate_visual_quality(self, frames: List[np.ndarray]) -> Dict:
        """Calculate visual quality metrics"""
        if not frames:
            return {}
        
        # Calculate average brightness, contrast, and sharpness
        brightness_scores = []
        contrast_scores = []
        sharpness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness (mean pixel value)
            brightness = np.mean(gray)
            brightness_scores.append(brightness)
            
            # Contrast (standard deviation of pixel values)
            contrast = np.std(gray)
            contrast_scores.append(contrast)
            
            # Sharpness (variance of Laplacian)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(sharpness)
        
        return {
            "avg_brightness": np.mean(brightness_scores),
            "avg_contrast": np.mean(contrast_scores),
            "avg_sharpness": np.mean(sharpness_scores),
            "brightness_consistency": 1.0 / (1.0 + np.var(brightness_scores)),
            "contrast_consistency": 1.0 / (1.0 + np.var(contrast_scores)),
            "sharpness_consistency": 1.0 / (1.0 + np.var(sharpness_scores))
        }
    
    def _calculate_fidelity_to_input(self, first_frame: np.ndarray, reference_image_path: str) -> float:
        """Calculate how well the first frame matches the input image"""
        reference_img = cv2.imread(reference_image_path)
        if reference_img is None:
            return 0.0
        
        # Resize reference to match first frame
        reference_resized = cv2.resize(reference_img, (first_frame.shape[1], first_frame.shape[0]))
        
        # Convert to grayscale
        first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        return ssim(first_frame_gray, reference_gray)
    
    def evaluate_all_videos(self, results_dir: str = "results") -> Dict:
        """Evaluate all videos in the results directory"""
        results_path = Path(results_dir)
        video_files = list(results_path.glob("*.mp4"))
        
        evaluations = {}
        
        for video_file in video_files:
            print(f"Evaluating {video_file.name}...")
            metrics = self.calculate_video_metrics(str(video_file))
            evaluations[video_file.name] = metrics
        
        return evaluations
    
    def create_comparison_report(self, evaluations: Dict, output_file: str = "evaluation_report.json"):
        """Create a comprehensive comparison report"""
        # Save detailed metrics
        with open(output_file, 'w') as f:
            json.dump(evaluations, f, indent=2)
        
        # Create summary comparison
        summary = self._create_summary_table(evaluations)
        
        # Generate visualizations
        self._create_comparison_plots(evaluations)
        
        return summary
    
    def _create_summary_table(self, evaluations: Dict) -> Dict:
        """Create a summary table comparing all models"""
        summary = {}
        
        for video_name, metrics in evaluations.items():
            if "error" in metrics:
                continue
                
            model_name = video_name.split('_')[0]  # Extract model name from filename
            
            summary[model_name] = {
                "temporal_consistency": metrics.get("temporal_consistency", 0),
                "motion_smoothness": metrics.get("motion_smoothness", 0),
                "avg_sharpness": metrics.get("visual_quality", {}).get("avg_sharpness", 0),
                "total_frames": metrics.get("total_frames", 0),
                "duration": metrics.get("duration", 0),
                "fidelity_to_input": metrics.get("fidelity_to_input", 0)
            }
        
        return summary
    
    def _create_comparison_plots(self, evaluations: Dict):
        """Create visualization plots for model comparison"""
        # Extract metrics for plotting
        model_names = []
        temporal_consistency = []
        motion_smoothness = []
        visual_quality = []
        
        for video_name, metrics in evaluations.items():
            if "error" in metrics:
                continue
                
            model_names.append(video_name.split('_')[0])
            temporal_consistency.append(metrics.get("temporal_consistency", 0))
            motion_smoothness.append(metrics.get("motion_smoothness", 0))
            visual_quality.append(metrics.get("visual_quality", {}).get("avg_sharpness", 0))
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temporal consistency
        axes[0, 0].bar(model_names, temporal_consistency)
        axes[0, 0].set_title('Temporal Consistency')
        axes[0, 0].set_ylabel('SSIM Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Motion smoothness
        axes[0, 1].bar(model_names, motion_smoothness)
        axes[0, 1].set_title('Motion Smoothness')
        axes[0, 1].set_ylabel('Smoothness Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Visual quality
        axes[1, 0].bar(model_names, visual_quality)
        axes[1, 0].set_title('Visual Quality (Sharpness)')
        axes[1, 0].set_ylabel('Sharpness Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall comparison radar chart would go here
        axes[1, 1].text(0.5, 0.5, 'Overall Comparison\n(Radar Chart)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison plots saved to results/model_comparison.png")

if __name__ == "__main__":
    evaluator = VideoEvaluator()
    
    # Evaluate all videos
    evaluations = evaluator.evaluate_all_videos()
    
    # Create comparison report
    summary = evaluator.create_comparison_report(evaluations)
    
    print("\n=== Model Evaluation Summary ===")
    for model, metrics in summary.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")