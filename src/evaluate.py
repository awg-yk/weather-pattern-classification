import argparse

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import build_model
from src.train import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WeatherMapDataset(args.data_dir, args.labels, transform=get_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = build_model(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=LABELS))


if __name__ == "__main__":
    main()
