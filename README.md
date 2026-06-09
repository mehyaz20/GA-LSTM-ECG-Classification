# GA-LSTM: Autonomous ECG Arrhythmia Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Paper

**"A Novel Autonomous GA-LSTM Framework for Cost-Sensitive 
ECG Arrhythmia Classification"**  
Mehdi Yazdani Paraei, Meysam Yadollahzadeh Tabari  
Department of Computer Engineering, Babol Branch, 
Islamic Azad University, Babol, Iran.

---

## Results Summary

| Dataset   | Accuracy | Weighted F1 | Macro F1 |
|-----------|----------|-------------|----------|
| MIT-BIH   | 98.34%   | 98.31%      | 90.43%   |
| PTB       | 99.42%   | 99.41%      | 99.27%   |

### Ablation Study (MIT-BIH)

| Model                  | Tuning Method  | Weighted F1 |
|------------------------|---------------|-------------|
| Base LSTM              | Manual        | 94.02%      |
| **GA-LSTM (Proposed)** | **Autonomous**| **98.31%**  |

### Per-Class F1-Score (MIT-BIH)

| Class              | Base LSTM | GA-LSTM |
|--------------------|-----------|---------|
| N (Normal)         | 99.01%    | 99.23%  |
| S (Supraventricular)| 74.88%   | 81.72%  |
| V (Ventricular)    | 91.95%    | 95.17%  |
| F (Fusion)         | 64.91%    | 77.42%  |
| Q (Unclassifiable) | 96.93%    | 98.63%  |

---

## GA Configuration

| Parameter         | Value            |
|-------------------|-----------------|
| Population Size   | 8               |
| Generations       | 2               |
| Crossover Type    | Two-point       |
| Mutation Rate     | 25%             |
| Selection         | Tournament      |
| Fitness Function  | Weighted F1     |
| Search Space      | hidden: {32,64,128}, lr: {1e-4,1e-3,5e-3}, dropout: {0.1,0.2,0.3}, batch: {128,256} |

### Optimal Hyperparameters (GA Output)

| Parameter     | Value |
|---------------|-------|
| Hidden Size   | 64    |
| Learning Rate | 0.001 |
| Dropout Rate  | 0.2   |
| Batch Size    | 256   |

---

## Datasets

| Dataset  | Samples | Classes | Source |
|----------|---------|---------|--------|
| MIT-BIH  | 109,446 | 5       | [PhysioNet](https://www.physionet.org/content/mitdb/) |
| PTB      | 14,552  | 2       | [PhysioNet](https://www.physionet.org/content/ptbdb/) |

Download from Kaggle:
- [MIT-BIH + PTB](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

Place CSV files in: `data/`

---

## Requirements

```bash
pip install torch torchvision pandas numpy scikit-learn \
            imbalanced-learn matplotlib seaborn
```

## Usage

```bash
# Step 1: GA Optimization (finds best hyperparameters)
python ga_optimization.py

# Step 2: Final Training on MIT-BIH
python train_mitbih.py

# Step 3: Final Training on PTB
python train_ptb.py
```

---

## Model Architecture

**MIT-BIH:** Unidirectional LSTM (2 layers, hidden=64, dropout=0.2)  
**PTB:** Bidirectional LSTM + Temporal Attention (hidden=128, dropout=0.3)

Both models trained with:
- Optimizer: AdamW with Cosine Annealing LR scheduler
- Loss: CrossEntropyLoss
- Hardware: Apple M3 Pro (MPS acceleration)

---

## Confusion Matrices

**MIT-BIH:**  
![MIT-BIH Confusion Matrix](results/cm_mitbih.png)

**PTB:**  
![PTB Confusion Matrix](results/cm_ptb_v2.png)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@article{yazdani2026galstm,
  title={A Novel Autonomous GA-LSTM Framework for 
         Cost-Sensitive ECG Arrhythmia Classification},
  author={Yazdani Paraei, Mehdi and 
          Yadollahzadeh Tabari, Meysam},
  journal={Under Review},
  year={2026}
}
```
