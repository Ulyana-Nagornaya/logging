# crf_llm.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class BertCrf(nn.Module):
    """BERT + CRF model for Named Entity Recognition."""
    def __init__(self, num_labels, bert_name="bert-base-uncased", dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        emissions = self.fc(self.dropout(sequence_output))

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.bool())
            return -log_likelihood
        else:
            return self.crf.decode(emissions, mask=attention_mask.bool())

    def save_to(self, path):
        torch.save(self.state_dict(), path)

    def load_from(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def compute_f1(preds, labels):
    """Compute micro-F1 score across predictions."""
    preds_flat = [tag for seq in preds for tag in seq]
    labels_flat = [label for seq in labels for label in seq if label != -100]
    return f1_score(labels_flat, preds_flat, average='micro')


def evaluate_model(model, val_loader, device):
    """Evaluate model performance on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            predictions = model(input_ids, attention_mask)
            all_preds.extend(predictions)

            active_labels = [
                [label for label, mask in zip(seq_labels, seq_mask) if mask]
                for seq_labels, seq_mask in zip(labels.cpu().numpy(), attention_mask.cpu().numpy())
            ]
            all_labels.extend(active_labels)

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss, all_preds, all_labels


def visualize_confusion_matrix(true_labels, pred_labels, id2label):
    """Plot confusion matrix excluding 'O' label."""
    filtered_true = []
    filtered_pred = []

    for t, p in zip(true_labels, pred_labels):
        true_tag = id2label[t]
        pred_tag = id2label[p]
        if true_tag != 'O':
            filtered_true.append(true_tag)
            filtered_pred.append(pred_tag)

    labels = sorted(set(filtered_true + filtered_pred) - {'O'})

    cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def train_model(model, train_loader, val_loader, epochs=10, patience=3, lr=5e-5):
    """Train model with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_losses, val_losses, val_f1_scores = [], [], []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_preds, val_labels = evaluate_model(model, val_loader, device)
        val_f1 = compute_f1(val_preds, val_labels)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val micro-F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model.save_to('model.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return train_losses, val_losses, val_f1_scores


def test_model(model, test_loader, id2label):
    """Run final evaluation on test set and show classification report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_from('model.pt')
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask)
            all_preds.extend(predictions)

            active_labels = [
                [label for label, mask in zip(seq_labels, seq_mask) if mask]
                for seq_labels, seq_mask in zip(labels.cpu().numpy(), attention_mask.cpu().numpy())
            ]
            all_labels.extend(active_labels)

    true_labels = []
    pred_labels = []

    for labels_seq, preds_seq in zip(all_labels, all_preds):
        for true_tag_id, pred_tag_id in zip(labels_seq, preds_seq):
            if true_tag_id != -100:
                true_labels.append(id2label[true_tag_id])
                pred_labels.append(id2label[pred_tag_id])

    print(classification_report(true_labels, pred_labels, digits=4))
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    print(f"Micro-F1: {micro_f1:.4f}")

    visualize_confusion_matrix(true_labels, pred_labels, id2label)