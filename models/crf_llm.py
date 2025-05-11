import logging
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

label2id = {
    "O": 0,
    "B-Peop": 1, "I-Peop": 2,
    "B-Org": 3, "I-Org": 4,
    "B-Loc": 5, "I-Loc": 6,
    "B-Other": 7, "I-Other": 8
}
id2label = {v: k for k, v in label2id.items()}

class NERDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row['tokens']
        labels = row['bio_tags']
        
        # Tokenize and align labels
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        word_ids = tokenized.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(0)
            else:
                if word_id != previous_word_idx:
                    aligned_labels.append(self.label2id[labels[word_id]])
                else:
                    # Convert B- to I- for subsequent subtokens
                    orig_label = labels[word_id]
                    if orig_label.startswith("B-"):
                        entity_type = orig_label[2:]
                        aligned_labels.append(self.label2id[f"I-{entity_type}"])
                    else:
                        aligned_labels.append(self.label2id[orig_label])
                previous_word_idx = word_id
        
        tokenized["labels"] = torch.tensor(aligned_labels)
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["labels"]
        }

class BertCrfModel(nn.Module):
    def __init__(self, num_labels=len(label2id), bert_name="bert-base-uncased", dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        emissions = self.fc(self.dropout(sequence_output))
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        return self.crf.decode(emissions, mask=attention_mask.bool())
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=16, device="cuda"):
        logger.info(f"Starting training on device: {device}")
        
        # Создаем датасеты
        train_dataset = NERDataset(X_train, self.tokenizer, label2id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = NERDataset(X_val, self.tokenizer, label2id)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        self.to(device)
        best_loss = float('inf')
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            # Тренировка
            self.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                loss = self(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_train_loss:.4f}")
            
            # Валидация
            if val_loader:
                self.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)
                        
                        loss = self(input_ids, attention_mask, labels)
                        total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                logger.info(f"Validation loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    self.save_model("best_bert_crf_model.pt")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
                    
                if patience_counter >= patience:
                    logger.info("Early stopping triggered. Training stopped.")
                    break
            
            scheduler.step()
        
        logger.info("Training completed")
        return self
    
    def evaluate(self, X_test, y_test, batch_size=16, device="cuda"):
        logger.info("Starting evaluation")
        test_dataset = NERDataset(X_test, self.tokenizer, label2id)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.to(device)
        self.eval()
        
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Получаем предсказания
                predictions = self(input_ids, attention_mask)
                
                # Обрабатываем метки
                active_labels = [
                    [label for label, mask in zip(seq_labels, seq_mask) if mask]
                    for seq_labels, seq_mask in zip(labels.cpu().numpy(), attention_mask.cpu().numpy())
                ]
                
                all_true.extend(active_labels)
                all_preds.extend(predictions)
        
        # Преобразуем в читаемые метки
        true_labels = [[id2label[tag] for tag in seq] for seq in all_true]
        pred_labels = [[id2label[tag] for tag in seq] for seq in all_preds]
        
        # Вычисляем метрики
        true_flat = [tag for seq in true_labels for tag in seq]
        pred_flat = [tag for seq in pred_labels for tag in seq]
        
        report = classification_report(true_flat, pred_flat, digits=4)
        logger.info(f"Classification report:\n{report}")
        
        micro_f1 = f1_score(true_flat, pred_flat, average='micro')
        weighted_f1 = f1_score(true_flat, pred_flat, average='weighted')
        
        logger.info(f"Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
        
        return {
            "classification_report": report,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "true_labels": true_labels,
            "pred_labels": pred_labels
        }
    
    def save_model(self, path):
        logger.info(f"Saving model to {path}")
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, path, device="cuda"):
        logger.info(f"Loading model from {path}")
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        return model