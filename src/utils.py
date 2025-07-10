import torch
import yaml
from diffusers.utils import export_to_video

def load_config(config_path='configs/global_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_device():
    config = load_config()
    device = torch.device(config['device'] if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float16 if config['precision'] == 'float16' else torch.float32
    return device, dtype

def optimize_pipeline(pipe, config):
    if config['memory_opts']['enable_cpu_offload']:
        pipe.enable_model_cpu_offload()
    if config['memory_opts']['enable_slicing']:
        pipe.vae.enable_slicing()
    if config['memory_opts']['enable_tiling']:
        pipe.vae.enable_tiling()
    return pipe

def save_video(frames, output_path, fps=8):
    export_to_video(frames, output_path, fps=fps)
    print(f"Video saved to {output_path}")