"""Run GrainAD on one five-view PNG without training data or DTD textures."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from models import backbones
from models.simplenet import SimpleNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ica', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('prediction.npz'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/models/simplenet.yaml'))
    config.update(device=torch.device(args.device), input_shape=[3, 224, 224],
                  ica_path=str(args.ica), add_ica=True, five_split=True, visualization=False)
    model = SimpleNet(torch.device(args.device))
    model.load(backbone=backbones.load('wideresnet50'), **config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.discriminator.load_state_dict(checkpoint['discriminator'])
    model.pre_projection.load_state_dict(checkpoint['pre_projection'])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with Image.open(args.image) as source:
        source = source.convert('RGB')
        width, height = source.size
        if width < 5:
            raise ValueError('Image must contain five horizontally concatenated views')
        step = width // 5
        views = [transform(source.crop((i * step, 0, (i + 1) * step if i < 4 else width, height)))
                 .unsqueeze(0).to(args.device) for i in range(5)]
    scores, maps, _, _, theta = model._predict(views)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, score=np.asarray(scores), aligned_anomaly_map=np.asarray(maps),
                        theta=torch.stack(theta).cpu().numpy())
    print(json.dumps({'anomaly_score': float(scores[0]), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
