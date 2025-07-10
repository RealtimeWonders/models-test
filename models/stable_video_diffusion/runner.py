import torch
from diffusers import StableVideoDiffusionPipeline
from src.utils import setup_device, optimize_pipeline, save_video

class StableVideoDiffusionRunner:
    def __init__(self, global_config):
        self.config = global_config
        self.device, self.dtype = setup_device()
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=self.dtype,
            variant="fp16"
        ).to(self.device)
        self.pipe = optimize_pipeline(self.pipe, self.config)

    def generate_video(self, input_image_path):
        from PIL import Image
        image = Image.open(input_image_path).resize(self.config['default']['resolution'])
        generator = torch.manual_seed(42)
        frames = self.pipe(
            image, 
            num_frames=self.config['default']['num_frames'],
            num_inference_steps=self.config['default']['inference_steps'],
            generator=generator
        ).frames[0]
        return frames

    def train(self, dataset_path, epochs):
        print(f"Fine-tuning Stable Video Diffusion on {dataset_path} for {epochs} epochs.")
        # Implement LoRA-based training here (e.g., using PEFT library)
        # Example: from peft import LoraConfig; ...

    def save_video(self, frames, output_path):
        save_video(frames, output_path)