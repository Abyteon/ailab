import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================
# 配置
# ================
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================
# 数据
# ================
df = pd.read_csv("data.csv")  # 需要两列：sentence,label
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sentence"], df["label"], test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
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


train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ================
# 自定义模型
# ================
class MyBertClassifier(nn.Module):
    def _init_(self, bert_model_name, num_labels):
        super()._init_()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] 向量
        out = self.dropout(pooled_output)
        logits = self.classifier(out)
        return logits


num_labels = len(set(df["label"]))
model = MyBertClassifier(MODEL_NAME, num_labels).to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()


# ================
# 训练函数
# ================
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
        correct += torch.sum(preds == labels)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()

    acc = correct.double() / len(loader.dataset)
    return sum(losses) / len(losses), acc


# ================
# 验证函数
# ================
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
            correct += torch.sum(preds == labels)

            losses.append(loss.item())

    acc = correct.double() / len(loader.dataset)
    return sum(losses) / len(losses), acc


# ================
# 开始训练
# ================
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )
    val_loss, val_acc = eval_model(model, val_loader, device)

    print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Val loss:   {val_loss:.4f}, acc: {val_acc:.4f}")

# ================
# 保存模型
# ================
model_save_path = "custom_bert_model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# ================
# 推理示例
# ================
model.eval()
example_text = "我非常喜欢这款产品，质量很好！"
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
