import torch
import os
import json
import time
import gc
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional
import cv2
import psutil
import warnings
from memory_profiler import profile

# Suppress warnings
warnings.filterwarnings("ignore")

class LocalModelTester:
    def __init__(self, output_dir: str = "results", device: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure device for Apple Silicon
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("Using MPS (Metal Performance Shaders) for Apple Silicon")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("Using CUDA")
            else:
                self.device = "cpu"
                print("Using CPU")
        else:
            self.device = device
            
        self.results = []
        self.memory_stats = []
        
        # Set memory optimization flags
        if self.device == "mps":
            # Enable memory efficient attention for MPS
            torch.backends.mps.enable_efficient_attention = True
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return {
            "ram_used_gb": process.memory_info().rss / (1024**3),
            "ram_percent": process.memory_percent()
        }
    
    def _clear_memory(self):
        """Clear memory and garbage collect"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def test_stable_video_diffusion(self, image_path: str, model_id: str = "stabilityai/stable-video-diffusion-img2vid"):
        """Test Stable Video Diffusion model optimized for Apple Silicon"""
        print(f"Testing {model_id} on {self.device}...")
        
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            # Load model with optimizations
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "mps" else torch.float32,
                variant="fp16" if self.device != "mps" else None,
                use_safetensors=True
            )
            
            # Apply optimizations
            pipe = pipe.to(self.device)
            
            # Enable memory efficient attention
            pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            # Resize to supported dimensions
            image = image.resize((1024, 576))
            
            # Monitor memory before inference
            memory_before = self._get_memory_usage()
            
            start_time = time.time()
            
            # Generate video with conservative settings
            with torch.no_grad():
                frames = pipe(
                    image, 
                    num_frames=14,  # Reduced for memory efficiency
                    num_inference_steps=20,  # Reduced steps
                    decode_chunk_size=2,  # Process in smaller chunks
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).frames[0]
            
            inference_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            # Save video
            model_name = model_id.split("/")[-1]
            output_path = self.output_dir / f"{model_name}_{Path(image_path).stem}.mp4"
            self._save_frames_as_video(frames, output_path)
            
            # Clear memory
            del pipe
            self._clear_memory()
            
            result = {
                "model": model_id,
                "input_image": image_path,
                "output_video": str(output_path),
                "inference_time": inference_time,
                "num_frames": len(frames),
                "resolution": "1024x576",
                "device": self.device,
                "memory_before_gb": memory_before["ram_used_gb"],
                "memory_after_gb": memory_after["ram_used_gb"],
                "memory_peak_gb": memory_after["ram_used_gb"] - memory_before["ram_used_gb"],
                "success": True
            }
            
        except Exception as e:
            self._clear_memory()
            result = {
                "model": model_id,
                "input_image": image_path,
                "error": str(e),
                "device": self.device,
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def test_cogvideox_2b(self, image_path: str):
        """Test CogVideoX-2B model (memory efficient)"""
        print("Testing CogVideoX-2B...")
        
        try:
            from diffusers import CogVideoXImageToVideoPipeline
            
            model_id = "THUDM/CogVideoX-2b"
            
            # Load model with optimizations
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "mps" else torch.float32,
                use_safetensors=True
            )
            
            pipe = pipe.to(self.device)
            
            # Enable optimizations
            pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((720, 480))  # CogVideoX preferred resolution
            
            memory_before = self._get_memory_usage()
            start_time = time.time()
            
            # Generate video
            with torch.no_grad():
                video = pipe(
                    image=image,
                    prompt="A manga character coming to life with subtle animation",
                    num_inference_steps=20,
                    num_frames=16,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).frames[0]
            
            inference_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            # Save video
            output_path = self.output_dir / f"cogvideox_2b_{Path(image_path).stem}.mp4"
            self._save_frames_as_video(video, output_path)
            
            del pipe
            self._clear_memory()
            
            result = {
                "model": model_id,
                "input_image": image_path,
                "output_video": str(output_path),
                "inference_time": inference_time,
                "num_frames": len(video),
                "resolution": "720x480",
                "device": self.device,
                "memory_before_gb": memory_before["ram_used_gb"],
                "memory_after_gb": memory_after["ram_used_gb"],
                "memory_peak_gb": memory_after["ram_used_gb"] - memory_before["ram_used_gb"],
                "success": True
            }
            
        except Exception as e:
            self._clear_memory()
            result = {
                "model": "THUDM/CogVideoX-2b",
                "input_image": image_path,
                "error": str(e),
                "device": self.device,
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def test_animatediff(self, image_path: str):
        """Test AnimateDiff model"""
        print("Testing AnimateDiff...")
        
        try:
            from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
            
            # Load motion adapter
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
            
            # Load pipeline
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float16 if self.device != "mps" else torch.float32,
                use_safetensors=True
            )
            
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)
            
            # Enable optimizations
            pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_model_cpu_offload'):
                pipe.enable_model_cpu_offload()
            
            # Load image and create prompt
            image = Image.open(image_path).convert("RGB")
            prompt = "A manga character with subtle movements, high quality animation"
            
            memory_before = self._get_memory_usage()
            start_time = time.time()
            
            # Generate video
            with torch.no_grad():
                video = pipe(
                    prompt=prompt,
                    num_frames=16,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).frames[0]
            
            inference_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            # Save video
            output_path = self.output_dir / f"animatediff_{Path(image_path).stem}.mp4"
            self._save_frames_as_video(video, output_path)
            
            del pipe, adapter
            self._clear_memory()
            
            result = {
                "model": "AnimateDiff",
                "input_image": image_path,
                "output_video": str(output_path),
                "inference_time": inference_time,
                "num_frames": len(video),
                "resolution": "512x512",
                "device": self.device,
                "memory_before_gb": memory_before["ram_used_gb"],
                "memory_after_gb": memory_after["ram_used_gb"],
                "memory_peak_gb": memory_after["ram_used_gb"] - memory_before["ram_used_gb"],
                "success": True
            }
            
        except Exception as e:
            self._clear_memory()
            result = {
                "model": "AnimateDiff",
                "input_image": image_path,
                "error": str(e),
                "device": self.device,
                "success": False
            }
            
        self.results.append(result)
        return result
    
    def test_tooncrafter(self, image_path: str):
        """Test ToonCrafter model (specialized for cartoon/manga animation)"""
        print("Testing ToonCrafter...")
        
        try:
            # For now, ToonCrafter doesn't have a direct diffusers integration
            # This is a placeholder implementation that would need the actual ToonCrafter code
            # from their GitHub repository: https://github.com/ToonCrafter/ToonCrafter
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 320))  # ToonCrafter's native resolution
            
            memory_before = self._get_memory_usage()
            start_time = time.time()
            
            # Placeholder implementation
            # In a real implementation, you would:
            # 1. Load ToonCrafter model from their GitHub repo
            # 2. Create a duplicate of the input image as end frame
            # 3. Generate interpolated frames with text prompt
            # 4. Return the video frames
            
            # For now, we'll simulate the expected behavior
            result = {
                "model": "Doubiiu/ToonCrafter",
                "input_image": image_path,
                "status": "Model implementation pending - requires manual installation from GitHub",
                "note": "ToonCrafter specializes in cartoon interpolation (2 frames -> 16 frame video)",
                "resolution": "512x320",
                "expected_frames": 16,
                "expected_duration": "2 seconds",
                "device": self.device,
                "memory_before_gb": memory_before["ram_used_gb"],
                "github_repo": "https://github.com/ToonCrafter/ToonCrafter",
                "success": False,
                "reason": "Manual installation required"
            }
            
            print("⚠️  ToonCrafter requires manual installation from GitHub")
            print("   Repository: https://github.com/ToonCrafter/ToonCrafter")
            print("   This model is specialized for cartoon/manga animation!")
            
        except Exception as e:
            self._clear_memory()
            result = {
                "model": "Doubiiu/ToonCrafter",
                "input_image": image_path,
                "error": str(e),
                "device": self.device,
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
            if isinstance(frame, Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                # Handle tensor frames
                frame_arrays.append(frame)
            
        # Create video writer
        height, width = frame_arrays[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_array in frame_arrays:
            if len(frame_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_array
            out.write(frame_bgr.astype(np.uint8))
            
        out.release()
        print(f"Video saved to {output_path}")
    
    def evaluate_results(self):
        """Evaluate and compare model results"""
        print("\n=== Local Model Evaluation Results ===")
        
        successful_models = []
        failed_models = []
        
        for result in self.results:
            if result['success']:
                successful_models.append(result)
                print(f"\n✅ {result['model']}:")
                print(f"  ⏱️  Inference time: {result['inference_time']:.2f}s")
                print(f"  🎬 Frames: {result['num_frames']}")
                print(f"  📐 Resolution: {result['resolution']}")
                print(f"  🖥️  Device: {result['device']}")
                print(f"  💾 Memory peak: {result['memory_peak_gb']:.2f}GB")
                print(f"  📁 Output: {result['output_video']}")
            else:
                failed_models.append(result)
                print(f"\n❌ {result['model']}:")
                print(f"  Error: {result['error']}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  ✅ Successful: {len(successful_models)}")
        print(f"  ❌ Failed: {len(failed_models)}")
        
        if successful_models:
            avg_time = np.mean([r['inference_time'] for r in successful_models])
            avg_memory = np.mean([r['memory_peak_gb'] for r in successful_models])
            print(f"  ⏱️  Average inference time: {avg_time:.2f}s")
            print(f"  💾 Average memory usage: {avg_memory:.2f}GB")
    
    def save_results(self, filename: str = "local_model_results.json"):
        """Save results to JSON file"""
        results_path = self.output_dir / filename
        
        # Add system info
        system_info = {
            "device": self.device,
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "total_ram_gb": psutil.virtual_memory().total / (1024**3)
        }
        
        full_results = {
            "system_info": system_info,
            "results": self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    def run_all_tests(self, image_path: str):
        """Run all available local model tests"""
        print(f"Testing all local models with image: {image_path}")
        print(f"Device: {self.device}")
        print(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        
        # Test models in order of memory efficiency
        test_functions = [
            ("Stable Video Diffusion", lambda: self.test_stable_video_diffusion(image_path)),
            ("CogVideoX-2B", lambda: self.test_cogvideox_2b(image_path)),
            ("AnimateDiff", lambda: self.test_animatediff(image_path)),
            ("ToonCrafter", lambda: self.test_tooncrafter(image_path))
        ]
        
        for model_name, test_func in test_functions:
            print(f"\n{'='*50}")
            print(f"Testing {model_name}")
            print(f"{'='*50}")
            
            try:
                test_func()
            except Exception as e:
                print(f"Failed to test {model_name}: {e}")
                
            # Clear memory between tests
            self._clear_memory()
            time.sleep(2)  # Brief pause
        
        # Evaluate and save results
        self.evaluate_results()
        self.save_results()

if __name__ == "__main__":
    # Example usage
    tester = LocalModelTester()
    
    # Test with a sample image
    input_dir = Path("data/input")
    if input_dir.exists():
        image_files = list(input_dir.glob("*.{jpg,jpeg,png,webp}"))
        if image_files:
            tester.run_all_tests(str(image_files[0]))
        else:
            print("No images found in data/input/. Please add manga images to test.")
    else:
        print("Please create data/input/ directory and add manga images to test.")