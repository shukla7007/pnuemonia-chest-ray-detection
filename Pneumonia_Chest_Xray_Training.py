# Pneumonia Chest X-ray Classification (ResNet18, no downloads) + graphs
# Dataset root must contain train/ val/ test (or chest_xray/train|val|test)
# Outputs saved to: /Users/anshulshukla/python/Computer vision/X ray classification

import os, sys, json, subprocess, traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- PATHS ----------
DATA_ROOT = "/Users/anshulshukla/Downloads/chest_xray"  # <- your dataset root
OUT_DIR   = "/Users/anshulshukla/python/Computer vision/X ray classification"
os.makedirs(OUT_DIR, exist_ok=True)

# Detect layout (train/val/test either directly or under chest_xray/)
CANDIDATES = [
    (os.path.join(DATA_ROOT, "train"),
     os.path.join(DATA_ROOT, "val"),
     os.path.join(DATA_ROOT, "test")),
    (os.path.join(DATA_ROOT, "chest_xray", "train"),
     os.path.join(DATA_ROOT, "chest_xray", "val"),
     os.path.join(DATA_ROOT, "chest_xray", "test")),
]
for tr, va, te in CANDIDATES:
    if os.path.isdir(tr) and os.path.isdir(va) and os.path.isdir(te):
        TRAIN_DIR, VAL_DIR, TEST_DIR = tr, va, te
        break
else:
    raise SystemExit("❌ Could not find train/val/test under DATA_ROOT.")

STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_PREFIX = os.path.join(OUT_DIR, f"cxr_{STAMP}")

# ---------- CONFIG ----------
INPUT_SIZE  = (224, 224)   # CPU-friendly
BATCH_SIZE  = 16
EPOCHS      = 8
LR          = 1e-4
NUM_WORKERS = 0            # macOS safe
PERSISTENT  = False

# ---------- HELPERS ----------
def save_and_open(path: str):
    ap = os.path.abspath(path)
    print(f"📁 Saved: {ap}")
    if sys.platform == "darwin":
        try: subprocess.run(["open", ap], check=False)
        except Exception: pass

def smoke_test():
    txt = os.path.join(OUT_DIR, "cxr_write_check.txt")
    with open(txt, "w") as f: f.write("Write OK\n")
    save_and_open(txt)
    plt.figure(); plt.plot([0,1,2],[0,1,0]); plt.title("CXR path test")
    png = os.path.join(OUT_DIR, "cxr_path_test.png")
    plt.savefig(png, bbox_inches="tight"); plt.close()
    save_and_open(png)

def build_loaders():
    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))  # force 3-ch
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE), to_rgb,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(INPUT_SIZE), to_rgb,
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tf_train)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=tf_eval)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=tf_eval)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, persistent_workers=PERSISTENT)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, persistent_workers=PERSISTENT)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, persistent_workers=PERSISTENT)
    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl

def build_model(num_classes: int, device):
    model = models.resnet18(weights=None)           # no internet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_epoch(model, dl, crit, opt, device):
    model.train(); tot = cor = 0; loss_sum = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); out = model(x); loss = crit(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
        cor += (out.argmax(1)==y).sum().item()
        tot += y.size(0)
    return (loss_sum/max(tot,1)), (cor/max(tot,1))

@torch.no_grad()
def eval_epoch(model, dl, crit, device):
    model.eval(); tot = cor = 0; loss_sum = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x); loss = crit(out, y)
        loss_sum += loss.item()*x.size(0)
        cor += (out.argmax(1)==y).sum().item()
        tot += y.size(0)
    return (loss_sum/max(tot,1)), (cor/max(tot,1))

def plot_curves(hist, prefix):
    xs = np.arange(1, len(hist["train_loss"])+1)
    # loss
    plt.figure()
    plt.plot(xs, hist["train_loss"], label="train")
    plt.plot(xs, hist["val_loss"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    p = f"{prefix}_loss.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)
    # acc
    plt.figure()
    plt.plot(xs, hist["train_acc"], label="train")
    plt.plot(xs, hist["val_acc"],   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy per Epoch")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    p = f"{prefix}_acc.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)

@torch.no_grad()
def full_eval(model, dl, classes, device):
    model.eval(); yp, yt = [], []
    for x, y in dl:
        x = x.to(device); out = model(x)
        yp.extend(out.argmax(1).cpu().tolist()); yt.extend(y.tolist())
    report = classification_report(yt, yp, target_names=classes, output_dict=True)
    cm = confusion_matrix(yt, yp)
    return report, cm

def save_report(report, prefix):
    j = f"{prefix}_report.json"; t = f"{prefix}_report.txt"
    with open(j, "w") as f: json.dump(report, f, indent=2)
    lines = ["Per-class (precision/recall/f1/support):\n"]
    for k, v in report.items():
        if k in ("accuracy","macro avg","weighted avg"): continue
        if isinstance(v, dict) and {"precision","recall","f1-score","support"} <= set(v.keys()):
            lines.append(f"{k:>12s}: P={v['precision']:.3f} R={v['recall']:.3f} "
                         f"F1={v['f1-score']:.3f} N={int(v['support'])}")
    lines.append(f"\naccuracy   : {report['accuracy']:.3f}")
    for name in ("macro avg","weighted avg"):
        v = report[name]
        lines.append(f"{name:>12s}: P={v['precision']:.3f} R={v['recall']:.3f} "
                     f"F1={v['f1-score']:.3f} N={int(v['support'])}")
    with open(t, "w") as f: f.write("\n".join(lines))
    save_and_open(j); save_and_open(t)

def plot_confusion(cm, classes, prefix):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right"); plt.yticks(ticks, classes)
    thresh = cm.max()/2.0 if cm.max()>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i,j])
            plt.text(j, i, str(val), ha="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    p = f"{prefix}_cm.png"; plt.savefig(p, bbox_inches="tight"); plt.close(); save_and_open(p)

def main():
    # quick write test
    smoke_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    train_ds, val_ds, test_ds, train_dl, val_dl, test_dl = build_loaders()
    classes = train_ds.classes
    print("Classes:", classes)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = build_model(len(classes), device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    try:
        for ep in range(1, EPOCHS+1):
            tr_loss, tr_acc = train_epoch(model, train_dl, crit, opt, device)
            va_loss, va_acc = eval_epoch(model,  val_dl,   crit,     device)
            history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc);  history["val_acc"].append(va_acc)
            print(f"Epoch {ep}/{EPOCHS} - loss {tr_loss:.4f}/{va_loss:.4f}  acc {tr_acc:.4f}/{va_acc:.4f}")
    except Exception:
        print("❌ Training error:"); traceback.print_exc()

    # save model + curves + csv
    mdl = f"{OUT_PREFIX}.pt"; torch.save(model.state_dict(), mdl); save_and_open(mdl)
    with open(f"{OUT_PREFIX}_history.csv","w") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},"
                    f"{history['train_acc'][i]:.6f},{history['val_acc'][i]:.6f}\n")
    save_and_open(f"{OUT_PREFIX}_history.csv")
    plot_curves(history, OUT_PREFIX)

    # final test evaluation
    try:
        report, cm = full_eval(model, test_dl, classes, device)
        save_report(report, OUT_PREFIX)
        plot_confusion(cm, classes, OUT_PREFIX)
    except Exception:
        print("❌ Test evaluation error:"); traceback.print_exc()

    print("\n✅ All outputs saved under:")
    print(os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()

