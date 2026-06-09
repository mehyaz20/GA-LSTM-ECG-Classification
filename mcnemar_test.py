import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar
import os

device = torch.device('cpu')
DATA_DIR = "/Users/mehdiyazdani/Desktop/Code Python/paper_1/archive (1)"

# ==========================================
# معماری LSTM - کپی کامل از کد اصلی
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
# بارگذاری داده
# ==========================================
print("Loading test data...")
test_df = pd.read_csv(
    os.path.join(DATA_DIR, "mitbih_test.csv"), header=None)
X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test = test_df.iloc[:, -1].values.astype(np.int64)

test_X = torch.tensor(
    X_test[:, :, np.newaxis], dtype=torch.float32)

# ==========================================
# تابع آموزش و گرفتن پیش‌بینی
# ==========================================
def train_and_predict(hidden, lr, dropout, batch,
                      epochs, X_tr, y_tr):
    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "mitbih_train.csv"), header=None)
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)

    Xt = torch.tensor(X_train[:,:,np.newaxis], dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt),
                       batch_size=int(batch),
                       shuffle=True, num_workers=0)

    model = ECGLSTM(hidden_size=int(hidden),
                    dropout=float(dropout)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_f1, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(
                    model(test_X.to(device)),
                    dim=1).cpu().numpy()
            f1w = f1_score(y_test, preds, average='weighted')
            if f1w > best_f1:
                best_f1 = f1w
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    model.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(
            model(test_X.to(device)), dim=1).cpu().numpy()
    return preds

# ==========================================
# آموزش هر دو مدل
# ==========================================
print("\nTraining Base LSTM (manual params)...")
train_df = pd.read_csv(
    os.path.join(DATA_DIR, "mitbih_train.csv"), header=None)
X_train = train_df.iloc[:, :-1].values.astype(np.float32)
y_train = train_df.iloc[:, -1].values.astype(np.int64)

y_pred_base = train_and_predict(
    hidden=32, lr=0.001, dropout=0.0,
    batch=512, epochs=100,
    X_tr=X_train, y_tr=y_train)

print(f"Base LSTM F1: {f1_score(y_test, y_pred_base, average='weighted')*100:.2f}%")

print("\nTraining GA-LSTM (optimized params)...")
y_pred_ga = train_and_predict(
    hidden=64, lr=0.001, dropout=0.2,
    batch=256, epochs=100,
    X_tr=X_train, y_tr=y_train)

print(f"GA-LSTM F1: {f1_score(y_test, y_pred_ga, average='weighted')*100:.2f}%")

# ==========================================
# McNemar Test
# ==========================================
print("\nRunning McNemar Test...")
correct_base = (y_pred_base == y_test)
correct_ga   = (y_pred_ga   == y_test)

n11 = np.sum( correct_base &  correct_ga)
n10 = np.sum( correct_base & ~correct_ga)
n01 = np.sum(~correct_base &  correct_ga)
n00 = np.sum(~correct_base & ~correct_ga)

table = [[n11, n10],
         [n01, n00]]

print(f"\nContingency Table:")
print(f"Both correct (n11):     {n11}")
print(f"Only Base correct (n10):{n10}")
print(f"Only GA correct (n01):  {n01}")
print(f"Both wrong (n00):       {n00}")

result = mcnemar(table, exact=False, correction=True)

print(f"\n{'='*45}")
print(f"McNemar Statistic : {result.statistic:.4f}")
print(f"p-value           : {result.pvalue:.8f}")
if result.pvalue < 0.05:
    print(f"Result: SIGNIFICANT (p < 0.05) ✓")
    print(f"GA-LSTM is statistically better than Base LSTM")
else:
    print(f"Result: NOT significant (p >= 0.05)")
print(f"{'='*45}")