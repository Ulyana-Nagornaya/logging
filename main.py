import logging
from models.crf_ml import CRFModel
from models.crf_llm import BertCrfModel
from utils import load_data, prepare_data
import torch

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_experiment(model, model_name, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    logging.info(f"[{model_name}] Starting training...")

    if isinstance(model, CRFModel):
        model.train(X_train, y_train)
    elif isinstance(model, BertCrfModel):
        model.train_model(X_train, y_train, X_val, y_val)

    logging.info(f"[{model_name}] Starting evaluation...")
    metrics = model.evaluate(X_test, y_test)

    logging.info(f"[{model_name}] Results:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {key}:")
            for k, v in value.items():
                logging.info(f"{k}: {v:.4f}")
        else:
            logging.info(f"{key}: {value:.4f}")

    logging.info(f"[{model_name}] Micro-F1 score: {metrics.get('accuracy', 0.0):.4f}")
    return metrics

def main():
    logging.info("Starting experiments: NER for Knowledge Graph Construction")

    logging.info("Loading CoNLL04 dataset...")
    df_train, df_test = load_data()
    
    # Загружаем валидационный датасет отдельно
    from datasets import load_dataset
    dataset = load_dataset("DFKI-SLT/conll04")
    df_val = dataset["validation"].to_pandas()

    # logging.info("Preparing data for CRF model...")
    # df_train_crf, X_crf_train, y_crf_train = prepare_data(df_train)
    # df_test_crf, X_crf_test, y_crf_test = prepare_data(df_test)
    
    # logging.info("Initializing CRF model...")
    # crf_model = CRFModel()
    # run_experiment(crf_model, "CRF", X_crf_train, y_crf_train, X_crf_test, y_crf_test)

    logging.info("Preparing data for BERT+CRF model...")
    # Используем prepare_data для BERT+CRF (добавит bio_tags)
    df_train_bert, X_bert_train, y_bert_train = prepare_data(df_train)  # Теперь bio_tags есть
    df_test_bert, X_bert_test, y_bert_test = prepare_data(df_test)
    df_val_bert, X_bert_val, y_bert_val = prepare_data(df_val)

    logging.info("Initializing BERT+CRF model...")
    bert_crf_model = BertCrfModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(bert_crf_model, "BERT+CRF", X_bert_train, y_bert_train, X_bert_test, y_bert_test, X_bert_val, y_bert_val)
if __name__ == "__main__":
    main()
    