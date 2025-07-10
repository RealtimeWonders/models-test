import torch
import os
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Any
from diffusers import StableVideoDiffusionPipeline
from transformers import pipeline
import cv2

class ModelTester:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = []
        
    def test_stable_video_diffusion(self, image_path: str, model_id: str = "stabilityai/stable-video-diffusion-img2vid"):
        """Test Stable Video Diffusion model"""
        print(f"Testing {model_id}...")
        
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
            
            image = Image.open(image_path).convert("RGB")
            image = image.resize((1024, 576))
            
            start_time = time.time()
            frames = pipe(image, num_frames=25, num_inference_steps=25).frames[0]
            inference_time = time.time() - start_time
            
            # Save video
            output_path = self.output_dir / f"svd_{Path(image_path).stem}.mp4"
            self._save_frames_as_video(frames, output_path)
            
            result = {
                "model": model_id,
                "input_image": image_path,
                "output_video": str(output_path),
                "inference_time": inference_time,
                "num_frames": len(frames),
                "resolution": "1024x576",
                "success": True
            }
            
        except Exception as e:
            result = {
                "model": model_id,
                "input_image": image_path,
                "error": str(e),
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def test_hunyuan_video_i2v(self, image_path: str):
        """Test HunyuanVideo-I2V model"""
        print("Testing HunyuanVideo-I2V...")
        
        try:
            # Note: This would need the actual HunyuanVideo implementation
            # Placeholder for when the model becomes available
            result = {
                "model": "tencent/HunyuanVideo-I2V",
                "input_image": image_path,
                "status": "Model not yet implemented",
                "success": False
            }
            
        except Exception as e:
            result = {
                "model": "tencent/HunyuanVideo-I2V",
                "input_image": image_path,
                "error": str(e),
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def test_wan_i2v(self, image_path: str):
        """Test Wan2.1-I2V-14B-720P model"""
        print("Testing Wan2.1-I2V-14B-720P...")
        
        try:
            # Note: This would need the actual Wan-AI implementation
            # Placeholder for when the model becomes available
            result = {
                "model": "Wan-AI/Wan2.1-I2V-14B-720P",
                "input_image": image_path,
                "status": "Model not yet implemented",
                "success": False
            }
            
        except Exception as e:
            result = {
                "model": "Wan-AI/Wan2.1-I2V-14B-720P",
                "input_image": image_path,
                "error": str(e),
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def _save_frames_as_video(self, frames: List[Image.Image], output_path: Path, fps: int = 8):
        """Save frames as video file"""
        if not frames:
            return
            
        # Convert PIL Images to numpy arrays
        frame_arrays = []
        for frame in frames:
            frame_arrays.append(np.array(frame))
            
        # Create video writer
        height, width = frame_arrays[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_array in frame_arrays:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        print(f"Video saved to {output_path}")
    
    def evaluate_results(self):
        """Evaluate and compare model results"""
        print("\n=== Model Evaluation Results ===")
        
        for result in self.results:
            if result['success']:
                print(f"\n{result['model']}:")
                print(f"  ✓ Success")
                print(f"  ⏱️  Inference time: {result['inference_time']:.2f}s")
                print(f"  🎬 Frames: {result['num_frames']}")
                print(f"  📐 Resolution: {result['resolution']}")
                print(f"  📁 Output: {result['output_video']}")
            else:
                print(f"\n{result['model']}:")
                print(f"  ❌ Failed: {result.get('error', result.get('status', 'Unknown error'))}")
    
    def save_results(self, filename: str = "model_comparison.json"):
        """Save results to JSON file"""
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    def run_all_tests(self, image_path: str):
        """Run all available model tests"""
        print(f"Testing all models with image: {image_path}")
        
        # Test Stable Video Diffusion variants
        svd_models = [
            "stabilityai/stable-video-diffusion-img2vid",
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        ]
        
        for model_id in svd_models:
            try:
                self.test_stable_video_diffusion(image_path, model_id)
            except Exception as e:
                print(f"Failed to test {model_id}: {e}")
        
        # Test newer models (when available)
        self.test_hunyuan_video_i2v(image_path)
        self.test_wan_i2v(image_path)
        
        # Evaluate and save results
        self.evaluate_results()
        self.save_results()

if __name__ == "__main__":
    # Example usage
    tester = ModelTester()
    
    # Test with a sample image (user should place manga images in data/input/)
    input_dir = Path("data/input")
    if input_dir.exists():
        image_files = list(input_dir.glob("*.{jpg,jpeg,png,webp}"))
        if image_files:
            tester.run_all_tests(str(image_files[0]))
        else:
            print("No images found in data/input/. Please add manga images to test.")
    else:
        print("Please create data/input/ directory and add manga images to test.")