# train.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from tqdm import tqdm
import csv

from dataset.dataloader import ECGDataset, ECGDataset2
from dataset.augment import ECGAugmenter
from torch.utils.data import ConcatDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for ecg, labels in tqdm(dataloader, desc="Training"):
        ecg = ecg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, vq_loss = model(ecg)
        loss = criterion(logits, labels) + vq_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for ecg, labels in tqdm(dataloader, desc="Validation"):
        ecg = ecg.to(device)
        labels = labels.to(device)

        logits,_ = model(ecg)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc, f1


def main():
    # Load config
    with open("config/protoaf_chapman_1.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Set seed
    set_seed(cfg["training"]["seed"])

    # Device
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Augmentation (only for training)
    train_aug = ECGAugmenter(
        target_length=cfg["augmentation"]["target_length"],
        p_amplitude_scale=cfg["augmentation"]["p_amplitude_scale"],
        scale_range=cfg["augmentation"]["scale_range"],
        p_time_resize=cfg["augmentation"]["p_time_resize"],
        resize_mode_prob=cfg["augmentation"]["resize_mode_prob"],
        p_gaussian_noise=cfg["augmentation"]["p_gaussian_noise"],
        noise_std_range=cfg["augmentation"]["noise_std_range"],
        p_baseline_wander=cfg["augmentation"]["p_baseline_wander"],
        bw_amplitude_range=cfg["augmentation"]["bw_amplitude_range"],
        bw_freq_range=cfg["augmentation"]["bw_freq_range"],
        p_lead_mask=cfg["augmentation"]["p_lead_mask"],
        mask_range=cfg["augmentation"]["mask_range"],
        fs=cfg["augmentation"]["fs"],
        inplace=cfg["augmentation"]["inplace"]
    )

    # Datasets
    # if 'chapman' in cfg["data"]["train_csv"]:
    #     train_dataset = ECGDataset(
    #         csv_path=cfg["data"]["train_csv"],
    #         data_root_dir=cfg["data"]["data_root_dir"],
    #         transform=train_aug,
    #         oversample=cfg["training"]["oversample"],
    #         random_seed=cfg["training"]["seed"]
    #     )
    #     val_dataset = ECGDataset(
    #         csv_path=cfg["data"]["val_csv"],
    #         data_root_dir=cfg["data"]["data_root_dir"],
    #         transform=None,
    #         oversample=False
    #     )
    # if 'afdb' in cfg["data"]["train_csv"]:
    #     train_dataset = ECGDataset2(
    #         txt_path=cfg["data"]["train_csv"],
    #         data_root_dir=cfg["data"]["data_root_dir"],
    #         transform=train_aug
    #     )
    #     val_dataset = ECGDataset2(
    #         txt_path=cfg["data"]["val_csv"],
    #         data_root_dir=cfg["data"]["data_root_dir"],
    #     )
    train_datasets = []
    val_datasets = []

    train_datasets.append(ECGDataset(
        csv_path="./dataset_preprocess/chapman/fold_1_train.csv",
        data_root_dir="/home/oyxq/PythonProjects/dataset/Chapman/PTDB",
        transform=train_aug,
    ))
    val_datasets.append(ECGDataset(
        csv_path="./dataset_preprocess/chapman/test.csv",
        data_root_dir="/home/oyxq/PythonProjects/dataset/Chapman/PTDB",
        transform=None,
    ))

    train_datasets.append(ECGDataset2(
        txt_path="./dataset_preprocess/afdb/fold_1_train_files.txt",
        data_root_dir="/home/oyxq/PythonProjects/dataset/MIT-BIH-AF/data",
        transform=train_aug
    ))
    val_datasets.append(ECGDataset2(
        txt_path="./dataset_preprocess/afdb/test_files.txt",
        data_root_dir="/home/oyxq/PythonProjects/dataset/MIT-BIH-AF/data",
        transform=None,
    ))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)



    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

    # Model
    if cfg["model"]["name"] == 'mobilenet':
        from model.MobileNet import mobilenet_v1_1d
        model = mobilenet_v1_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'shufflenet':
        from model.ShuffleNet import shufflenet_v1_1d
        model = shufflenet_v1_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'efficientnet':
        from model.EfficientNet import efficientnet_1d
        model = efficientnet_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'mobilevit':
        from model.MobileViT import mobilevit_1d
        model = mobilevit_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'edgenext':
        from model.EdgeNext import edgenext_1d
        model = edgenext_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'levit':
        from model.LeViT import levit_1d
        model = levit_1d(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'fussingtransformer':
        from model.FussingTransformer import fussingtrans
        model = fussingtrans(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'afibrinet':
        from model.AFibriNet import afib_net
        model = afib_net(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'lcgnet':
        from model.LCGNet import lcgnet
        model = lcgnet(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'eplm':
        from model.EPLM import eplm
        model = eplm(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'cadnet':
        from model.CADNet import cadnet
        model = cadnet(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'ecgdg':
        from model.ECGDG import ECGDG
        model = ECGDG(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'rawecgnet':
        from model.RawECGNet import RawECGNet
        model = RawECGNet(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'upvcnet':
        from model.uPVCNet import uPVCNet
        model = uPVCNet(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    elif cfg["model"]["name"] == 'protoaf':
        from model.ProtoAF import ProtoAF
        model = ProtoAF(
            num_classes=cfg["model"]["num_classes"],
        ).to(device)
    else:
        print('model name is wrong!')

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # Training loop
    best_val_acc = 0.0
    history = []
    checkpoint_dir = os.path.dirname(cfg["training"]["checkpoint_path"])
    csv_path = os.path.splitext(cfg["training"]["checkpoint_path"])[0] + ".csv"

    for epoch in range(cfg["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['num_epochs']}")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "train_f1": round(train_f1, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_f1": round(val_f1, 6),
        })

        # Save best model
        if cfg["training"]["save_checkpoint"] and val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(cfg["training"]["checkpoint_path"]), exist_ok=True)
            torch.save(model.state_dict(), cfg["training"]["checkpoint_path"])
            print(f"✅ Best model saved with Val ACC: {best_val_acc:.4f}")

        os.makedirs(checkpoint_dir, exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(history[-1])

    print("Training finished.")


if __name__ == "__main__":
    main()