import torch
import torch.nn as nn

# ==========================================
# معماری MIT-BIH
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
# معماری PTB
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
        out, _ = self.lstm(x)
        score  = self.attn(out)
        weight = torch.softmax(score, dim=1)
        ctx    = (out * weight).sum(1)
        return self.fc(self.drop(ctx))

# ==========================================
# محاسبه
# ==========================================
def model_info(model, name, seq_len=187):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 * 1024)

    # inference time
    import time
    model.eval()
    x = torch.randn(1, seq_len, 1)
    # warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    # measure
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Parameters : {params:,}")
    print(f"Size       : {size_mb:.3f} MB")
    print(f"Inference  : {avg_ms:.3f} ms/sample")
    print(f"{'='*40}")

model_mitbih = ECGLSTM(hidden_size=64, dropout=0.2)
model_ptb    = ECGLSTMAttn(hidden=128, dropout=0.3)

model_info(model_mitbih, "GA-LSTM (MIT-BIH)")
model_info(model_ptb,    "BiLSTM+Attn (PTB)")