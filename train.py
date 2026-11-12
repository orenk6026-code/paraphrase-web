"""Fine-tune SBERT untuk deteksi parafrasa/kemiripan.
Format CSV: sentence1,sentence2,label  (label 1 = parafrasa/mirip makna, 0 = tidak)
Pakai: python train.py --data data/pairs.csv --epochs 4
Hasil: folder ./model-ft (+ threshold.txt). Lalu jalankan web dengan:
  set EMB_MODEL=model-ft  &&  set SEM_TH=<isi threshold.txt>  &&  python app.py
"""
import argparse, numpy as np, pandas as pd
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments, losses)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/pairs.csv")
ap.add_argument("--base", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
ap.add_argument("--out", default="model-ft")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
a = ap.parse_args()

df = pd.read_csv(a.data).dropna()
df["label"] = df["label"].astype(float)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n_dev = max(2, int(len(df) * 0.15))
dev, train = df[:n_dev], df[n_dev:]
print(f"train={len(train)} dev={len(dev)}")

model = SentenceTransformer(a.base)
train_ds = Dataset.from_dict({"sentence1": train.sentence1.tolist(),
                              "sentence2": train.sentence2.tolist(),
                              "score": train.label.tolist()})
evaluator = EmbeddingSimilarityEvaluator(dev.sentence1.tolist(), dev.sentence2.tolist(),
                                         dev.label.tolist(), name="dev")
args = SentenceTransformerTrainingArguments(
    output_dir="runs", num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
    learning_rate=a.lr, warmup_ratio=0.1, eval_strategy="epoch",
    save_strategy="no", logging_steps=20, report_to="none")
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                                     loss=losses.CosineSimilarityLoss(model), evaluator=evaluator)
print("Sebelum fine-tune:", evaluator(model))
trainer.train()
print("Sesudah fine-tune:", evaluator(model))
model.save(a.out)

# Kalibrasi ambang: cari cosine yang memaksimalkan F1 pada data dev
e1 = model.encode(dev.sentence1.tolist(), normalize_embeddings=True)
e2 = model.encode(dev.sentence2.tolist(), normalize_embeddings=True)
cos = (e1 * e2).sum(1)
y = dev.label.values >= 0.5
best = (0, 0.8)
for t in np.arange(0.3, 0.98, 0.01):
    p = cos >= t
    tp = (p & y).sum(); fp = (p & ~y).sum(); fn = (~p & y).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    if f1 > best[0]: best = (f1, float(t))
print(f"Ambang terbaik={best[1]:.2f}  F1 dev={best[0]:.3f}")
open("threshold.txt", "w").write(f"{best[1]:.2f}")
