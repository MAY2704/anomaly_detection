import sys
from pathlib import Path

# Ensure local package imports work when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf

from config import CFG
from data import generate_data
from preprocess import prepare_sequences, split_and_scale
from model import build_lstm_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    np.random.seed(CFG.NP_SEED)
    tf.random.set_seed(CFG.TF_SEED)

    df = generate_data(CFG.N_CP, CFG.N_MONTHS, seed=CFG.NP_SEED)
    X_all, Y_pred_all, labels_all = prepare_sequences(df, list(CFG.FEATURES), CFG.TIME_STEP)

    (X_train, X_test, Y_pred_train, Y_pred_test, lab_train, lab_test, scaler) = split_and_scale(
        X_all, Y_pred_all, labels_all, test_size=CFG.TEST_SIZE, random_state=CFG.NP_SEED
    )

    X_train_norm = X_train[lab_train == 0]
    Y_pred_train_norm = Y_pred_train[lab_train == 0]

    logger.info("Train normal sequences: %d, Test sequences: %d, True anomalies in test: %d",
                X_train_norm.shape[0], X_test.shape[0], int(np.sum(lab_test)))

    model = build_lstm_predictor(
        CFG.TIME_STEP,
        len(CFG.FEATURES),
        lstm_units=CFG.LSTM_UNITS,
        dense_units=CFG.DENSE_UNITS,
        learning_rate=CFG.LEARNING_RATE,
    )

    model.fit(X_train_norm, Y_pred_train_norm, epochs=CFG.EPOCHS, batch_size=CFG.BATCH_SIZE, verbose=1)

    n_val = max(1, int(len(X_train_norm) * CFG.VAL_FRAC))
    X_val = X_train_norm[:n_val]
    Y_pred_val = Y_pred_train_norm[:n_val]
    val_pred = model.predict(X_val, verbose=0)
    val_mse = np.mean((Y_pred_val - val_pred) ** 2, axis=(1, 2))

    test_pred = model.predict(X_test, verbose=0)
    test_mse = np.mean((Y_pred_test - test_pred) ** 2, axis=(1, 2))

    mms = MinMaxScaler().fit(val_mse.reshape(-1, 1))
    norm_score = mms.transform(test_mse.reshape(-1, 1)).ravel()

    # Option A: fixed threshold from config (0..1)
    y_pred = (norm_score > CFG.THRESHOLD).astype(int)
    # Option B (recommended): compute threshold from validation distribution, e.g. 95th percentile:
    # computed_threshold = np.percentile(val_mse, 95)
    # y_pred = (test_mse > computed_threshold).astype(int)

    y_true = lab_test
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    total_anomalies = FN + TP

    logger.info("--- Final Evaluation ---")
    logger.info("Threshold (cfg): %.4f", CFG.THRESHOLD)
    logger.info("Precision: %.4f, Recall: %.4f, F1: %.4f", p, r, f)
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Total True Anomalies (P): %d, FP: %d, FN: %d", total_anomalies, FP, FN)

if __name__ == "__main__":
    main()