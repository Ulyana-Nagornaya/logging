# crf_llm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm.auto import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Label mapping
label2id = {
    "O": 0,
    "B-Peop": 1,
    "I-Peop": 2,
    "B-Org": 3,
    "I-Org": 4,
    "B-Loc": 5,
    "I-Loc": 6,
    "B-Other": 7,
    "I-Other": 8
}
id2label = {v: k for k, v in label2id.items()}

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class NERDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]["tokens"]
        bio_tags = self.df.iloc[idx]["bio_tags"]

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        labels = []
        previous_word_idx = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_idx:
                labels.append(label2id[bio_tags[word_id]])
            else:
                labels.append(-100)
            previous_word_idx = word_id

        encoding["labels"] = torch.tensor(labels)
        return {key: val.squeeze(0) for key, val in encoding.items()}


# BERT + CRF Model
class BertCrf(nn.Module):
    def __init__(self, num_labels, bert_name="bert-base-uncased", dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        emissions = self.fc(self.dropout(outputs[0]))
        if labels is not None:
            return -self.crf(emissions, labels, mask=attention_mask.bool())
        return self.crf.decode(emissions, mask=attention_mask.bool())

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


# Model wrapper
class BertCrfModel:
    def __init__(self, num_labels=len(label2id), epochs=10, batch_size=17):
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = BertCrf(num_labels=self.num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, X_train, y_train):
        logger.info("Preparing training data...")
        df_train = pd.DataFrame({"tokens": X_train, "bio_tags": y_train})
        train_dataset = NERDataset(df_train, tokenizer, label2id)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")
            scheduler.step()

        logger.info("Training completed.")

    def evaluate(self, X_test, y_test):
        logger.info("Preparing test data...")
        df_test = pd.DataFrame({"tokens": X_test, "bio_tags": y_test})
        test_dataset = NERDataset(df_test, tokenizer, label2id)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                predictions = self.model(input_ids, attention_mask)

                active_labels = [
                    [label for label, mask in zip(seq_labels, seq_mask) if mask]
                    for seq_labels, seq_mask in zip(labels.cpu().numpy(), attention_mask.cpu().numpy())
                ]
                active_preds = predictions

                for label_seq, pred_seq in zip(active_labels, active_preds):
                    for l, p in zip(label_seq, pred_seq):
                        if l != -100:
                            true_labels.append(id2label[l])
                            pred_labels.append(id2label[p])

        logger.info("Classification Report:")
        report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
        print(classification_report(true_labels, pred_labels, digits=4))

        micro_f1 = f1_score(true_labels, pred_labels, average='micro')
        logger.info(f"Micro-F1 score: {micro_f1:.4f}")

        return {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score'],
            'accuracy': micro_f1,
            'classification_report': report
        }