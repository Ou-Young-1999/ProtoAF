# test.py

import os
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm
import numpy as np

from dataset.dataloader import ECGDataset,ECGDataset2


@torch.no_grad()
def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_ids = []

    for ecg, labels, seg_ids in tqdm(dataloader):
        ecg = ecg.to(device)
        labels = labels.to(device)

        logits,_ = model(ecg)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_ids.extend(seg_ids)

    return all_labels, all_preds, all_probs, all_ids


def main():
    # Load config
    with open("config/protoaf_chapman_1.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader (no augmentation, no oversampling)
    if 'chapman' in cfg["data"]["test_csv"]:
        test_dataset = ECGDataset(
            csv_path=cfg["data"]["test_csv"],
            data_root_dir=cfg["data"]["data_root_dir"],
            transform=None,
            oversample=False
        )
    if 'afdb' in cfg["data"]["test_csv"]:
        test_dataset = ECGDataset2(
            txt_path=cfg["data"]["test_csv"],
            data_root_dir=cfg["data"]["data_root_dir"],
            transform=None,
        )
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4)

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

    # Load checkpoint
    ckpt_path = cfg["training"]["checkpoint_path"]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded model from {ckpt_path}")

    # Test
    labels, preds, probs, ids = test_model(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    f1 = f1_score(labels, preds, average='macro')
    print(f"\nTest Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Class 0", "Class 1"],digits=4))

    # Optional: Saving
    import pandas as pd
    df = pd.DataFrame({
        "segment_id": ids,
        "true_label": labels,
        "pred_label": preds,
        "probs": probs
    })
    os.makedirs(cfg["training"]["result"], exist_ok=True)
    df.to_csv(os.path.join(cfg["training"]["result"],"test_predictions.csv"), index=False)
    print("\nPredictions saved to test_predictions.csv")

    with open(os.path.join(cfg["training"]["result"],"test_results.txt"), "w") as f:
        f.write("\nTest Results:\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Macro): {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"AUPRC: {auprc:.4f}")
        f.write("\nClassification Report:\n")
        report = classification_report(labels, preds, target_names=["Class 0", "Class 1"], digits=4)
        f.write(report)
        cm = confusion_matrix(labels, preds)
        f.write("\nConfusion Matrix (raw counts):\n")
        f.write(np.array2string(cm, separator=', ', prefix='  '))
        f.write("\n")
    print("\nResults saved to test_results.csv")


if __name__ == "__main__":
    main()