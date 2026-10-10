"""Save actual training augmentation examples, including Mixup and CutMix."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .augmentations import STATS, cutmix, mixup
from .datasets import loader, training_datasets
from ..utils import load_config, seed_everything

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='reports/generated/augmentations.png')
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config['seed'])
    dataset, _, _ = training_datasets(config)
    images, labels = next(iter(loader(dataset, config, True)))
    images, labels = images[:8], labels[:8]
    targets = torch.nn.functional.one_hot(labels, config['model']['num_classes']).float()
    settings = config['augmentation']
    alternatives = [images, mixup(images, targets, settings['mixup_alpha'])[0],
                    cutmix(images, targets, settings['cutmix_alpha'])[0]]
    mean, std = STATS[config['data']['dataset']]
    fig, axes = plt.subplots(3, len(images), figsize=(2 * len(images), 6), squeeze=False)
    for row, batch in enumerate(alternatives):
        for column, image in enumerate(batch):
            display = (image * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]).clamp(0, 1)
            axes[row, column].imshow(display.permute(1, 2, 0))
            axes[row, column].axis('off')
            if column == 0:
                axes[row, column].set_title(['Spatial augmentation', 'Mixup', 'CutMix'][row])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)

if __name__ == '__main__':
    main()
