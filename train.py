# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import os

from dataset import FakeNewsDataset
from model import FakeNewsModel
from graph import KnowledgeGraphBuilder

# ================= CONFIG =================
DATA_DIR = "dataset_fakenews"
MODEL_NAME = "distilbert-base-multilingual-cased"  # CPU-safe multilingual model
MAX_LEN = 48
BATCH_SIZE = 1
EPOCHS = 1          # keep 1–2 on CPU
LR = 2e-5
SAMPLE_SIZE = 500  # subset of FakeNewsNet
# =========================================

def train():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # -------- Load tokenizer & data --------
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    kg_builder = KnowledgeGraphBuilder()

    dataset = FakeNewsDataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        sample_size=SAMPLE_SIZE,
        kg_builder=kg_builder
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # -------- Initialize model --------
    print("Initializing model...")
    model = FakeNewsModel(model_name=MODEL_NAME, use_kg=True).to(device)

    # 🔴 IMPORTANT: optimize ONLY trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_acc = 0.0

    # ================= TRAINING =================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kg_features = batch["kg_features"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)

            outputs = model(input_ids, attention_mask, kg_features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Avg Train Loss: {avg_loss:.4f}")

        # ================= VALIDATION =================
        model.eval()
        preds, gold = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                kg_features = batch["kg_features"].to(device)
                labels = batch["label"].float().to(device)

                outputs = model(input_ids, attention_mask, kg_features)
                predictions = (torch.sigmoid(outputs) > 0.5).int()

                preds.extend(predictions.cpu().numpy().ravel())
                gold.extend(labels.cpu().numpy().ravel())

        acc = accuracy_score(gold, preds)
        f1 = f1_score(gold, preds)

        print(f"Val Accuracy: {acc:.4f} | Val F1: {f1:.4f}")

        # -------- Save best model --------
        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/fake_news_model.pt")
            print("✅ Model saved!")

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train()