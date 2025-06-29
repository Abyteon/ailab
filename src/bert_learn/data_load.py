# from datasets import load_dataset
from huggingface_hub.utils.tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers.utils.dummy_pt_objects import get_linear_schedule_with_warmup
# ds = load_dataset("clue/clue", "afqmc")
# print(ds)

df = pd.read_csv("./data.csv")

texts = df["sentence"].tolist()
labels = df[" label"].tolist()
print(texts)
print(labels)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sentence"], df[" label"], test_size=0.2, random_state=42
)
print("Train texts:", train_texts[:5])
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)


class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }


train_ds = MyDataset(train_texts, train_labels, tokenizer)
val_ds = MyDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)
# print("DataLoader created with batch size 32.")
# print(len(ds))  # Print the first item in the dataset to verify
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3
)


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
 L       labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        accuracy = correct_predictions.double() / len(data_loader.dataset)

    return total_loss / len(data_loader), accuracy.item()


for epoch in range(3):
    print(f"Epoch {epoch + 1}/{3}")
    train_loss, train_accuracy = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    valid_loss, valid_accuracy = train_epoch(
        model, val_loader, optimizer, scheduler, device
    )
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
    print(f"Validation Loss: {valid_loss}, Validation Accuracy: {valid_accuracy}")

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
