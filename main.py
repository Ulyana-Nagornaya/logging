import logging
from utils import load_conll04, prepare_dataframes, get_dataloaders
from crf_llm import BertCrf, train_model, test_model
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report, f1_score
import numpy as np
import torch

# Set up logging
logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_crf_experiment(model, X_train, y_train, X_test, y_test):
    """Run experiment with traditional CRF model."""
    logging.info("[CRF] Starting training...")
    model.fit(X_train, y_train)

    logging.info("[CRF] Starting evaluation...")
    y_pred = model.predict(X_test)

    # Flatten sequences for metrics
    y_test_flat = [item for seq in y_test for item in seq]
    y_pred_flat = [item for seq in y_pred for item in seq]

    report = classification_report(y_test_flat, y_pred_flat, output_dict=True, zero_division=0)
    micro_f1 = f1_score(y_test_flat, y_pred_flat, average='micro')

    logging.info("[CRF] Classification Results:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"  {label}:")
            for metric_name, value in metrics.items():
                logging.info(f"    {metric_name}: {value:.4f}")
        else:
            logging.info(f"{label}: {metrics:.4f}")

    logging.info(f"[CRF] Micro-F1 score: {micro_f1:.4f}")
    return {"classification_report": report, "micro_f1": micro_f1}


def run_bert_crf_experiment(model, train_loader, val_loader, test_loader):
    """Run experiment with BERT + CRF model."""
    logging.info("[BERT-CRF] Starting training...")
    train_model(model, train_loader, val_loader)

    logging.info("[BERT-CRF] Starting evaluation on test set...")
    test_metrics = test_model(model, test_loader)
    return test_metrics


def main():
    logging.info("Starting experiments: NER for Knowledge Graph Construction")

    # Step 1: Load dataset
    logging.info("Loading CoNLL04 dataset...")
    df_train, df_val, df_test = load_conll04()

    logging.info("Preparing data for CRF model...")
    label2id = {
        "O": 0, "B-Peop": 1, "I-Peop": 2,
        "B-Org": 3, "I-Org": 4,
        "B-Loc": 5, "I-Loc": 6,
        "B-Other": 7, "I-Other": 8
    }
    id2label = {v: k for k, v in label2id.items()}

    # Prepare BIO tags
    df_train, df_val, df_test = prepare_dataframes(df_train, df_val, df_test)

    # Extract tokens and labels for CRF (list of list of strings)
    X_crf_train = df_train['tokens'].tolist()
    y_crf_train = df_train['bio_tags'].tolist()
    X_crf_test = df_test['tokens'].tolist()
    y_crf_test = df_test['bio_tags'].tolist()

    # Run CRF Baseline
    logging.info("Initializing CRF model...")
    crf_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    run_crf_experiment(crf_model, X_crf_train, y_crf_train, X_crf_test, y_crf_test)

    # Step 2: Run BERT + CRF model
    logging.info("Preparing data for BERT-CRF model...")
    train_loader, val_loader, test_loader = get_dataloaders(df_train, df_val, df_test)

    logging.info("Initializing BERT-CRF model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_crf_model = BertCrf(num_labels=len(label2id)).to(device)

    run_bert_crf_experiment(bert_crf_model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()