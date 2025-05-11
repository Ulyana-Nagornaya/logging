from datasets import load_dataset
import spacy

# utils.py

from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchcrf import CRF
import torch
import torch.nn as nn


def load_conll04():
    """Load CoNLL04 dataset using Hugging Face datasets."""
    dataset = load_dataset("DFKI-SLT/conll04")
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return df_train, df_val, df_test


def generate_bio_tags(tokens, entities):
    """Generate BIO tags from entity annotations."""
    bio_tags = ['O'] * len(tokens)
    for entity in entities:
        start = entity['start']
        end = entity['end']
        typ = entity['type']
        if start < len(bio_tags):
            bio_tags[start] = f'B-{typ}'
            for i in range(start + 1, end):
                if i < len(bio_tags):
                    bio_tags[i] = f'I-{typ}'
    return bio_tags


def prepare_dataframes(df_train, df_val, df_test):
    """Apply BIO tagging to all splits and filter relevant columns."""
    df_train['bio_tags'] = df_train.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)
    df_val['bio_tags'] = df_val.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)
    df_test['bio_tags'] = df_test.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)

    df_train = df_train[['tokens', 'bio_tags']]
    df_val = df_val[['tokens', 'bio_tags']]
    df_test = df_test[['tokens', 'bio_tags']]
    return df_train, df_val, df_test


def get_label_mappings():
    """Return mappings between labels and IDs."""
    label2id = {
        "O": 0,
        "B-Peop": 1, "I-Peop": 2,
        "B-Org": 3, "I-Org": 4,
        "B-Loc": 5, "I-Loc": 6,
        "B-Other": 7, "I-Other": 8
    }
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def tokenize_and_align_labels(tokens, labels, tokenizer, label2id, max_length=128):
    """Tokenize tokens and align labels with subwords."""
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    aligned_labels = []

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(0)  # CLS/SEP/PAD
        else:
            original_label = labels[word_id]
            if word_id != previous_word_idx:
                aligned_labels.append(label2id[original_label])
            else:
                if original_label.startswith(("B-", "I-")):
                    entity_type = original_label[2:]
                    aligned_labels.append(label2id[f"I-{entity_type}"])
                else:
                    aligned_labels.append(label2id["O"])
            previous_word_idx = word_id

    tokenized_inputs["labels"] = torch.tensor(aligned_labels)
    return tokenized_inputs


class NERDataset(Dataset):
    """PyTorch Dataset for NER with BERT tokenizer compatibility."""
    def __init__(self, df, tokenizer, label2id, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]["tokens"]
        labels = self.df.iloc[idx]["bio_tags"]
        tokenized_data = tokenize_and_align_labels(tokens, labels, self.tokenizer, self.label2id, self.max_length)
        return {
            "input_ids": tokenized_data["input_ids"].squeeze(0),
            "attention_mask": tokenized_data["attention_mask"].squeeze(0),
            "labels": tokenized_data["labels"]
        }


def get_dataloaders(df_train, df_val, df_test, batch_size=17, max_length=128):
    """Create DataLoader instances for training, validation, and test sets."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label2id, _ = get_label_mappings()

    train_dataset = NERDataset(df_train, tokenizer, label2id, max_length)
    val_dataset = NERDataset(df_val, tokenizer, label2id, max_length)
    test_dataset = NERDataset(df_test, tokenizer, label2id, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

nlp = spacy.load("en_core_web_sm")


def load_data():
    dataset = load_dataset("DFKI-SLT/conll04")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return df_train, df_test

def add_pos_tags(row):
    tokens = row["tokens"]
    doc = nlp(" ".join(tokens))
    row["pos_tags"] = [token.pos_ for token in doc]
    return row

def generate_bio_tags(tokens, entities):
    bio_tags = ["O"] * len(tokens)
    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        typ = entity["type"]
        if start < len(bio_tags):
            bio_tags[start] = f"B-{typ}"
            for i in range(start + 1, end):
                if i < len(bio_tags):
                    bio_tags[i] = f"I-{typ}"
    return bio_tags

def get_char_ngrams(token, min_n=2, max_n=5):
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(token) - n + 1):
            ngrams.add(token[i:i + n])
    return ngrams

def extract_features(df_row, i):
    tokens = df_row["tokens"]
    pos_tags = df_row["pos_tags"]
    token = tokens[i]
    pos = pos_tags[i]
    features = {
        "bias": 1.0,
        "pos": pos,
        "word.lower()": token.lower(),
        "is_capitalized": token[0].isupper() if len(token) > 0 else False,
        "is_all_caps": token.isupper(),
        "word.is_punctuation": token in [".", ",", ";", "?", "!"],
        "prefix4": token[:4] if len(token) >= 4 else token,
        "suffix4": token[-4:] if len(token) >= 4 else token,
    }
    if i > 0:
        prev_token = tokens[i - 1]
        prev_pos = pos_tags[i - 1]
        features.update({
            "-1:word.lower()": prev_token.lower(),
            "-1:pos": prev_pos if len(prev_pos) >= 2 else prev_pos,
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        next_token = tokens[i + 1]
        next_pos = pos_tags[i + 1]
        features.update({
            "+1:word.lower()": next_token.lower(),
            "+1:pos": next_pos if len(next_pos) >= 2 else next_pos,
        })
    else:
        features["EOS"] = True
    return features

def sent2features(df_row):
    return [extract_features(df_row, i) for i in range(len(df_row["tokens"]))]

def sent2labels(df_row):
    return df_row["bio_tags"]

def prepare_data(df):
    df = df.apply(add_pos_tags, axis=1)
    df["bio_tags"] = df.apply(lambda row: generate_bio_tags(row["tokens"], row["entities"]), axis=1)
    X = df.apply(sent2features, axis=1).tolist()
    y = df.apply(sent2labels, axis=1).tolist()
    return  df, X, y
