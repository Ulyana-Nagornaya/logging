# main.py

import logging
from crf_llm import BertCrfModel  # Make sure crf_llm.py is in the same directory
from datasets import load_dataset
import pandas as pd

# Configure logging
logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_experiment(model, model_name, X_train, y_train, X_test, y_test):
    logging.info(f"[{model_name}] Starting training...")
    model.train(X_train, y_train)

    logging.info(f"[{model_name}] Starting evaluation...")
    metrics = model.evaluate(X_test, y_test)

    logging.info(f"[{model_name}] Results:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {key}:")
            for k, v in value.items():
                logging.info(f"    {k}: {v:.4f}")
        else:
            logging.info(f"  {key}: {value:.4f}")

    logging.info(f"[{model_name}] Micro-F1 score: {metrics.get('accuracy', 0.0):.4f}")
    return metrics

def load_and_prepare_data():
    """
    Load CoNLL04 dataset and prepare tokens and bio_tags
    """
    logging.info("Loading CoNLL04 dataset...")
    dataset = load_dataset("DFKI-SLT/conll04")

    def extract_tokens_and_tags(df):
        df = df.to_pandas()
        df = df[['tokens', 'bio_tags']]
        return df['tokens'].tolist(), df['bio_tags'].tolist()

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Combine train and validation for final training
    combined_train_df = pd.concat([train_df, val_df], ignore_index=True)

    X_train, y_train = extract_tokens_and_tags(combined_train_df)
    X_test, y_test = extract_tokens_and_tags(test_df)

    return X_train, y_train, X_test, y_test


def main():
    logging.info("Starting experiments: NER for Knowledge Graph Construction")

    # Load and prepare data
    X_train, y_train, X_test, y_test = load_and_prepare_data()

    # Initialize and run BERT + CRF model
    logging.info("Initializing BERT + CRF model...")
    bert_crf_model = BertCrfModel()

    run_experiment(bert_crf_model, "BERT+CRF", X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()