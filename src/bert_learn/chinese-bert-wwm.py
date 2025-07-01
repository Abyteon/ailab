import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
)
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

# from sklearn.model_selection import train_test_split
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.df = pd.read_csv(file_path)  # 需要两列：text,label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        label_id = self.df.loc[idx, "label_id"]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


class AnathorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


MODULE_NAME = "hfl/chinese-bert-wwm"  # 使用中文BERT模型
MAX_LEN = 128
BATCH_SIZE = 16
train_encoded_csv_file = (
    "../../data/train_encoded_thucnews_data.csv"  # 假设已经编码好的CSV文件
)
val_encoded_csv_file = (
    "../../data/val_encoded_thucnews_data.csv"  # 假设已经编码好的CSV文件
)


# df = pd.read_csv("data.csv")  # 需要两列：sentence,label

# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     df["sentence"], df["label"], test_size=0.2, random_state=42
# )

# print("Train texts:", train_texts, train_labels)
# print("Val texts:", val_texts, val_labels)

tokenizer = BertTokenizer.from_pretrained(MODULE_NAME)

# train_dataset = MyDataset(train_texts, train_labels, tokenizer, MAX_LEN)
# val_dataset = MyDataset(val_texts, val_labels, tokenizer, MAX_LEN)
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_dataset = MyDataset(train_encoded_csv_file, tokenizer, MAX_LEN)
val_dataset = MyDataset(val_encoded_csv_file, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MyBertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        out = self.dropout(pooled_output)
        logits = self.classifier(out)
        return logits


EPOCHS = 3  # 假设训练3个epoch
# num_classes = len(set(train_labels))  # 根据标签数量设置输出类别数
num_classes = 10  # 假设有10个类别
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MyBertClassifier(MODULE_NAME, num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS  # 假设训练3个epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=2, num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct = 0

    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels, dtype=torch.float)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

    acc = correct / len(loader.dataset)
    return sum(losses) / len(losses), acc


def eval_model(model, loader, device):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels, dtype=torch.float)

            losses.append(loss.item())

    acc = correct / len(loader.dataset)
    return sum(losses) / len(losses), acc


for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    val_loss, val_acc = eval_model(model, val_loader, device)

    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val loss:   {val_loss:.4f}, acc: {val_acc:.4f}")

model_save_path = "my_bert_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

model.eval()
example_text = "这是一个测试句子。"

encoding = tokenizer(
    example_text,
    return_tensors="pt",
    max_length=MAX_LEN,
    truncation=True,
    padding="max_length",
)
input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)

print("预测概率:", probs.cpu())
print("预测类别:", pred.item())
