import argparse
from models.stable_video_diffusion.runner import StableVideoDiffusionRunner
from models.animatediff.runner import AnimateDiffRunner
from models.tooncrafter.runner import ToonCrafterRunner
from src.utils import load_config

MODEL_REGISTRY = {
    'stable_video_diffusion': StableVideoDiffusionRunner,
    'animatediff': AnimateDiffRunner,
    'tooncrafter': ToonCrafterRunner,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODEL_REGISTRY.keys(), help='Model to use')
    parser.add_argument('--input', default='data/input/image.png', help='Input image/path')
    parser.add_argument('--output', default='data/output/video.mp4', help='Output video path')
    parser.add_argument('--train', action='store_true', help='Run training instead of generation')
    parser.add_argument('--dataset_path', help='Dataset for training')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    args = parser.parse_args()

    config = load_config()
    runner_class = MODEL_REGISTRY[args.model]
    runner = runner_class(config)

    if args.train:
        runner.train(args.dataset_path, args.epochs)
    else:
        video = runner.generate_video(args.input)
        runner.save_video(video, args.output)

if __name__ == '__main__':
    main()