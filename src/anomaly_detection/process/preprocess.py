import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_sequences(df, feats, ts):
    data = []
    for _, g in df.groupby("CP_ID"):
        vals = g[feats].values
        labels = g["IS_TRUE_ANOMALY"].values
        for i in range(len(vals) - ts - 1):
            data.append((vals[i : i + ts], vals[i + 1 : i + ts + 1], labels[i + ts]))
    if not data:
        return np.empty((0, ts, len(feats))), np.empty((0, ts, len(feats))), np.empty((0,))
    X_all, Y_pred_all, labels_all = zip(*data)
    return np.array(X_all), np.array(Y_pred_all), np.array(labels_all)

def split_and_scale(X_all, Y_pred_all, labels_all, test_size=0.6, random_state=42):
    X_train_seq, X_test_seq, Y_pred_train_seq, Y_pred_test_seq, lab_train, lab_test = train_test_split(
        X_all, Y_pred_all, labels_all, test_size=test_size, random_state=random_state, stratify=labels_all
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train_seq.reshape(-1, X_train_seq.shape[-1]))

    def scale_data(X):
        return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    X_train, X_test = scale_data(X_train_seq), scale_data(X_test_seq)
    Y_pred_train, Y_pred_test = scale_data(Y_pred_train_seq), scale_data(Y_pred_test_seq)

    return (X_train, X_test, Y_pred_train, Y_pred_test, lab_train, lab_test, scaler)