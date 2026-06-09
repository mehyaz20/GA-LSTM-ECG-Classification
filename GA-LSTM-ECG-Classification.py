import os, gc, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. تنظیمات
# ==========================================
device = torch.device('cpu')
print(f"Device: {device}")

MITBIH_DIR = "/Users/mehdiyazdani/Desktop/Code Python/paper_1/archive (1)"
PTB_DIR    = "/Users/mehdiyazdani/Desktop/Code Python/paper_1/archive (1)"

# ==========================================
# 2. معماری LSTM
# ==========================================
class ECGLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2,
                 dropout=0.2, num_classes=5):
        super().__init__()
        h = int(hidden_size)
        d = float(dropout)
        self.lstm = nn.LSTM(1, h, int(num_layers),
                           batch_first=True,
                           dropout=d if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(d)
        self.fc   = nn.Linear(h, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(self.drop(hn[-1]))

# ==========================================
# 3. DataLoader
# ==========================================
def make_loader(X, y, batch_size, shuffle=True):
    Xt = torch.tensor(X[:, :, np.newaxis], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(Xt, yt),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )

# ==========================================
# 4. تابع آموزش کامل
# ==========================================
def full_train(name, X_tr, y_tr, X_te, y_te,
               hidden, lr, dropout, batch,
               num_classes=5, epochs=150):

    print(f"\n{'='*55}\n{name}\n{'='*55}")

    test_X = torch.tensor(
        X_te[:, :, np.newaxis], dtype=torch.float32)

    loader    = make_loader(X_tr, y_tr, int(batch))
    model     = ECGLSTM(hidden_size=int(hidden),
                        dropout=float(dropout),
                        num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=float(lr), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_f1, best_state = 0.0, None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(
                    model(test_X.to(device)),
                    dim=1).cpu().numpy()
            f1w  = f1_score(y_te, preds, average='weighted')
            loss_avg = total_loss / len(loader)
            history.append((epoch+1, loss_avg, f1w))
            print(f"Epoch [{epoch+1:3d}/{epochs}]  "
                  f"Loss: {loss_avg:.4f}  "
                  f"F1: {f1w*100:.2f}%")
            if f1w > best_f1:
                best_f1 = f1w
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # لود بهترین مدل
    model.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        final_preds = torch.argmax(
            model(test_X.to(device)),
            dim=1).cpu().numpy()

    acc = accuracy_score(y_te, final_preds)
    f1w = f1_score(y_te, final_preds, average='weighted')
    print(f"\n>>> BEST  Acc={acc*100:.2f}%  F1={f1w*100:.2f}%")

    # آزاد کردن حافظه
    model.cpu()
    del model, optimizer, loader
    gc.collect()

    return y_te, final_preds, acc, f1w, history

# ==========================================
# 5. MIT-BIH Dataset
# ==========================================
print("\n" + "="*55)
print("DATASET 1: MIT-BIH Arrhythmia Database")
print("="*55)

train_df = pd.read_csv(
    os.path.join(MITBIH_DIR, "mitbih_train.csv"), header=None)
test_df  = pd.read_csv(
    os.path.join(MITBIH_DIR, "mitbih_test.csv"),  header=None)

X_train_mb = train_df.iloc[:, :-1].values.astype(np.float32)
y_train_mb = train_df.iloc[:, -1].values.astype(np.int64)
X_test_mb  = test_df.iloc[:, :-1].values.astype(np.float32)
y_test_mb  = test_df.iloc[:, -1].values.astype(np.int64)

del train_df, test_df
gc.collect()

print(f"Train: {X_train_mb.shape} | Test: {X_test_mb.shape}")
print(f"Class dist (train): {np.bincount(y_train_mb)}")

CLASS_NAMES_MB = [
    'N (Normal)', 'S (Supraventricular)',
    'V (Ventricular)', 'F (Fusion)', 'Q (Unclassifiable)'
]

# پارامترهای بهینه GA از مرحله قبل
H, L, D, B = 64, 0.001, 0.2, 256

y_true_mb, y_pred_mb, acc_mb, f1_mb, hist_mb = full_train(
    "GA-LSTM — MIT-BIH (Final, 150 epochs)",
    X_train_mb, y_train_mb,
    X_test_mb,  y_test_mb,
    hidden=H, lr=L, dropout=D, batch=B,
    num_classes=5, epochs=150
)

print(f"\n{'='*55}")
print("MIT-BIH CLASSIFICATION REPORT")
print(f"{'='*55}")
print(classification_report(
    y_true_mb, y_pred_mb,
    target_names=CLASS_NAMES_MB, digits=4))

# Confusion Matrix MIT-BIH
cm_mb = confusion_matrix(y_true_mb, y_pred_mb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mb, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES_MB,
            yticklabels=CLASS_NAMES_MB)
plt.title('Confusion Matrix: GA-LSTM on MIT-BIH')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cm_mitbih.png', dpi=300)
print("Saved: cm_mitbih.png")

# Learning Curve MIT-BIH
epochs_h, losses_h, f1s_h = zip(*hist_mb)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs_h, losses_h, 'b-o', markersize=4)
ax1.set_title('Training Loss — MIT-BIH')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax2.plot(epochs_h, [f*100 for f in f1s_h], 'g-o', markersize=4)
ax2.set_title('Validation F1-Score — MIT-BIH')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 (%)')
ax2.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_mitbih.png', dpi=300)
print("Saved: learning_curve_mitbih.png")

# حافظه آزاد کن
del X_train_mb, y_train_mb, X_test_mb, y_test_mb
gc.collect()

# ==========================================
# 6. PTB Dataset
# ==========================================
print("\n" + "="*55)
print("DATASET 2: PTB Diagnostic ECG Database")
print("="*55)

# PTB دو کلاس داره: Normal=1, Abnormal=0
ptb_normal   = pd.read_csv(
    os.path.join(PTB_DIR, "ptbdb_normal.csv"),   header=None)
ptb_abnormal = pd.read_csv(
    os.path.join(PTB_DIR, "ptbdb_abnormal.csv"), header=None)

print(f"PTB Normal:   {ptb_normal.shape}")
print(f"PTB Abnormal: {ptb_abnormal.shape}")

ptb_df = pd.concat([ptb_normal, ptb_abnormal],
                   ignore_index=True)
del ptb_normal, ptb_abnormal
gc.collect()

X_ptb = ptb_df.iloc[:, :-1].values.astype(np.float32)
y_ptb = ptb_df.iloc[:, -1].values.astype(np.int64)
del ptb_df
gc.collect()

print(f"PTB Total: {X_ptb.shape}")
print(f"Class dist: {np.bincount(y_ptb)}")

# Train/Test split 80/20
from sklearn.model_selection import train_test_split
X_train_ptb, X_test_ptb, y_train_ptb, y_test_ptb = \
    train_test_split(X_ptb, y_ptb,
                     test_size=0.2,
                     random_state=42,
                     stratify=y_ptb)
del X_ptb, y_ptb
gc.collect()

print(f"PTB Train: {X_train_ptb.shape} | "
      f"Test: {X_test_ptb.shape}")

CLASS_NAMES_PTB = ['Abnormal', 'Normal']

y_true_ptb, y_pred_ptb, acc_ptb, f1_ptb, hist_ptb = full_train(
    "GA-LSTM — PTB Diagnostic (Final, 150 epochs)",
    X_train_ptb, y_train_ptb,
    X_test_ptb,  y_test_ptb,
    hidden=H, lr=L, dropout=D, batch=B,
    num_classes=2, epochs=150
)

print(f"\n{'='*55}")
print("PTB CLASSIFICATION REPORT")
print(f"{'='*55}")
print(classification_report(
    y_true_ptb, y_pred_ptb,
    target_names=CLASS_NAMES_PTB, digits=4))

# Confusion Matrix PTB
cm_ptb = confusion_matrix(y_true_ptb, y_pred_ptb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ptb, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES_PTB,
            yticklabels=CLASS_NAMES_PTB)
plt.title('Confusion Matrix: GA-LSTM on PTB')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('cm_ptb.png', dpi=300)
print("Saved: cm_ptb.png")

# Learning Curve PTB
epochs_p, losses_p, f1s_p = zip(*hist_ptb)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs_p, losses_p, 'b-o', markersize=4)
ax1.set_title('Training Loss — PTB')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax2.plot(epochs_p, [f*100 for f in f1s_p], 'g-o', markersize=4)
ax2.set_title('Validation F1-Score — PTB')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 (%)')
ax2.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_ptb.png', dpi=300)
print("Saved: learning_curve_ptb.png")

# ==========================================
# 7. خلاصه نهایی هر دو dataset
# ==========================================
print(f"\n{'='*55}")
print(f"{'FINAL SUMMARY — BOTH DATASETS':^55}")
print(f"{'='*55}")
print(f"MIT-BIH  →  Acc={acc_mb*100:.2f}%  "
      f"Weighted F1={f1_mb*100:.2f}%")
print(f"PTB      →  Acc={acc_ptb*100:.2f}%  "
      f"Weighted F1={f1_ptb*100:.2f}%")
print(f"{'='*55}")
print("\nFiles saved:")
print("  cm_mitbih.png")
print("  cm_ptb.png")
print("  learning_curve_mitbih.png")
print("  learning_curve_ptb.png")
