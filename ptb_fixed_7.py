import os, gc, torch, torch.nn as nn, torch.optim as optim
import pandas as pd, numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
PTB_DIR = "/Users/mehdiyazdani/Desktop/Code Python/paper_1/archive (1)"

# ==========================================
# بارگذاری و آماده‌سازی
# ==========================================
ptb_normal   = pd.read_csv(os.path.join(PTB_DIR, "ptbdb_normal.csv"),   header=None)
ptb_abnormal = pd.read_csv(os.path.join(PTB_DIR, "ptbdb_abnormal.csv"), header=None)
ptb_df = pd.concat([ptb_normal, ptb_abnormal], ignore_index=True)
del ptb_normal, ptb_abnormal

X = ptb_df.iloc[:, :-1].values.astype(np.float32)
y = ptb_df.iloc[:, -1].values.astype(np.int64)
del ptb_df
gc.collect()

# نرمال‌سازی - مهم برای PTB
X_mean = X.mean(axis=1, keepdims=True)
X_std  = X.std(axis=1, keepdims=True) + 1e-8
X = (X - X_mean) / X_std

print(f"PTB class dist: {np.bincount(y)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
del X, y
gc.collect()

# ==========================================
# WeightedRandomSampler به جای SMOTE
# ==========================================
class_counts = np.bincount(y_tr)
sample_weights = np.array([1.0/class_counts[c] for c in y_tr])
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(y_tr),
    replacement=True
)

train_X = torch.tensor(X_tr[:, :, np.newaxis], dtype=torch.float32)
train_y = torch.tensor(y_tr, dtype=torch.long)
test_X  = torch.tensor(X_te[:, :, np.newaxis], dtype=torch.float32)

loader = DataLoader(
    TensorDataset(train_X, train_y),
    batch_size=64,
    sampler=sampler,
    num_workers=0
)

# ==========================================
# معماری با Attention - برای PTB بهتر کار می‌کنه
# ==========================================
class ECGLSTMAttn(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden*2, 2)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, 2H)
        score  = self.attn(out)         # (B, T, 1)
        weight = torch.softmax(score, dim=1)
        ctx    = (out * weight).sum(1)  # (B, 2H)
        return self.fc(self.drop(ctx))

model     = ECGLSTMAttn(hidden=128, dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),
                        lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(loader),
    epochs=100,
    pct_start=0.3
)

# ==========================================
# آموزش
# ==========================================
print("\n" + "="*55)
print("GA-LSTM+Attention — PTB")
print("="*55)

best_f1, best_state = 0.0, None
history = []

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(
                model(test_X.to(device)),
                dim=1).cpu().numpy()
        f1w = f1_score(y_te, preds, average='weighted')
        f1m = f1_score(y_te, preds, average='macro')
        print(f"Epoch [{epoch+1:3d}/100]  "
              f"Loss: {total_loss/len(loader):.4f}  "
              f"F1w: {f1w*100:.2f}%  "
              f"F1macro: {f1m*100:.2f}%")
        history.append((epoch+1, total_loss/len(loader), f1w))
        if f1m > best_f1:   # macro F1 برای balance بهتره
            best_f1 = f1m
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

# ==========================================
# ارزیابی نهایی
# ==========================================
model.load_state_dict(
    {k: v.to(device) for k, v in best_state.items()})
model.eval()
with torch.no_grad():
    preds = torch.argmax(
        model(test_X.to(device)), dim=1).cpu().numpy()

acc = accuracy_score(y_te, preds)
f1w = f1_score(y_te, preds, average='weighted')
f1m = f1_score(y_te, preds, average='macro')
print(f"\n>>> BEST  Acc={acc*100:.2f}%  "
      f"F1w={f1w*100:.2f}%  F1macro={f1m*100:.2f}%")

CLASS_NAMES_PTB = ['Abnormal', 'Normal']
print("\n" + "="*55)
print(classification_report(y_te, preds,
      target_names=CLASS_NAMES_PTB, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_te, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES_PTB,
            yticklabels=CLASS_NAMES_PTB)
plt.title('Confusion Matrix: GA-LSTM on PTB')
plt.tight_layout()
plt.savefig('cm_ptb_v2.png', dpi=300)

# Learning Curve
if history:
    ep, lo, fi = zip(*history)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, lo, 'b-o', markersize=4)
    a1.set_title('Training Loss — PTB')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Loss'); a1.grid(True)
    a2.plot(ep, [f*100 for f in fi], 'g-o', markersize=4)
    a2.set_title('Validation F1 — PTB')
    a2.set_xlabel('Epoch'); a2.set_ylabel('F1 (%)'); a2.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve_ptb_v2.png', dpi=300)

print("Saved: cm_ptb_v2.png | learning_curve_ptb_v2.png")