"""
Memory optimization configuration for M3 Max MacBook Pro (48GB RAM)
"""
import torch
import os
from typing import Dict, Any

class MemoryConfig:
    """Memory optimization settings for Apple Silicon"""
    
    def __init__(self, total_ram_gb: float = 48.0):
        self.total_ram_gb = total_ram_gb
        self.device = self._get_optimal_device()
        self.config = self._get_memory_config()
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for inference"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_memory_config(self) -> Dict[str, Any]:
        """Get memory optimization configuration"""
        base_config = {
            # PyTorch settings
            "torch_dtype": torch.float32 if self.device == "mps" else torch.float16,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            
            # Model loading
            "variant": None if self.device == "mps" else "fp16",
            "revision": "main",
            
            # Attention optimization
            "enable_attention_slicing": True,
            "attention_slice_size": "auto",
            "enable_cpu_offload": True,
            
            # Memory management
            "max_memory_gb": self.total_ram_gb * 0.8,  # Use 80% of available RAM
            "cpu_offload_threshold": 0.6,  # Offload when 60% memory used
            
            # Inference settings
            "decode_chunk_size": 2,
            "enable_vae_slicing": True,
            "enable_sequential_cpu_offload": True,
        }
        
        # Device-specific optimizations
        if self.device == "mps":
            base_config.update({
                "mps_fallback": True,
                "enable_efficient_attention": True,
                "batch_size": 1,  # Conservative for MPS
                "num_inference_steps": 20,  # Reduced for faster inference
                "guidance_scale": 7.5,
            })
        elif self.device == "cuda":
            base_config.update({
                "enable_flash_attention": True,
                "batch_size": 2,
                "num_inference_steps": 25,
                "guidance_scale": 8.0,
            })
        else:  # CPU
            base_config.update({
                "batch_size": 1,
                "num_inference_steps": 15,
                "guidance_scale": 6.0,
            })
        
        return base_config
    
    def setup_environment(self):
        """Setup environment variables for optimal performance"""
        env_vars = {
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
        }
        
        if self.device == "mps":
            env_vars.update({
                "PYTORCH_MPS_ALLOCATOR_POLICY": "garbage_collection",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            })
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def optimize_model(self, model):
        """Apply memory optimizations to model"""
        if hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing()
        
        if hasattr(model, 'enable_vae_slicing'):
            model.enable_vae_slicing()
        
        if hasattr(model, 'enable_model_cpu_offload') and self.config["enable_cpu_offload"]:
            model.enable_model_cpu_offload()
        
        if hasattr(model, 'enable_sequential_cpu_offload') and self.config["enable_sequential_cpu_offload"]:
            model.enable_sequential_cpu_offload()
        
        # Set memory efficient settings
        if hasattr(model, 'set_progress_bar_config'):
            model.set_progress_bar_config(disable=True)
        
        return model
    
    def get_inference_params(self, model_type: str = "default") -> Dict[str, Any]:
        """Get optimized inference parameters for different model types"""
        base_params = {
            "num_inference_steps": self.config["num_inference_steps"],
            "guidance_scale": self.config["guidance_scale"],
            "generator": torch.Generator(device=self.device).manual_seed(42),
        }
        
        if model_type == "stable_video_diffusion":
            base_params.update({
                "num_frames": 14,  # Reduced for memory efficiency
                "decode_chunk_size": self.config["decode_chunk_size"],
                "motion_bucket_id": 127,
                "fps": 7,
                "noise_aug_strength": 0.02,
            })
        elif model_type == "cogvideox":
            base_params.update({
                "num_frames": 16,
                "height": 480,
                "width": 720,
            })
        elif model_type == "animatediff":
            base_params.update({
                "num_frames": 16,
                "height": 512,
                "width": 512,
            })
        elif model_type == "tooncrafter":
            base_params.update({
                "num_frames": 16,
                "height": 320,
                "width": 512,
                "fps": 8,
                "prompt": "smooth animation, cartoon style, manga character movement",
                "interpolation_steps": 16,  # ToonCrafter specific
                "cartoon_style": True,
            })
        
        return base_params
    
    def get_model_loading_params(self) -> Dict[str, Any]:
        """Get optimized model loading parameters"""
        return {
            "torch_dtype": self.config["torch_dtype"],
            "low_cpu_mem_usage": self.config["low_cpu_mem_usage"],
            "use_safetensors": self.config["use_safetensors"],
            "variant": self.config["variant"],
            "revision": self.config["revision"],
        }
    
    def clear_memory(self):
        """Clear device memory"""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "ram_used_gb": memory_info.rss / (1024**3),
            "ram_percent": process.memory_percent(),
            "available_gb": (psutil.virtual_memory().available) / (1024**3)
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=== Memory Configuration ===")
        print(f"Device: {self.device}")
        print(f"Total RAM: {self.total_ram_gb}GB")
        print(f"Max Memory Usage: {self.config['max_memory_gb']}GB")
        print(f"Torch dtype: {self.config['torch_dtype']}")
        print(f"Attention slicing: {self.config['enable_attention_slicing']}")
        print(f"CPU offload: {self.config['enable_cpu_offload']}")
        print(f"Decode chunk size: {self.config['decode_chunk_size']}")
        print("=" * 30)

# Global memory configuration instance
memory_config = MemoryConfig()

# Setup environment on import
memory_config.setup_environment()

# Utility functions
def get_optimized_model_params():
    """Get optimized model loading parameters"""
    return memory_config.get_model_loading_params()

def get_inference_params(model_type: str = "default"):
    """Get optimized inference parameters"""
    return memory_config.get_inference_params(model_type)

def optimize_model(model):
    """Apply memory optimizations to model"""
    return memory_config.optimize_model(model)

def clear_memory():
    """Clear device memory"""
    memory_config.clear_memory()

def get_memory_usage():
    """Get current memory usage"""
    return memory_config.get_memory_usage()

def print_memory_config():
    """Print current memory configuration"""
    memory_config.print_config()

if __name__ == "__main__":
    # Test the configuration
    print_memory_config()
    print("\nCurrent memory usage:", get_memory_usage())