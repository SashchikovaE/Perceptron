import numpy as np
from train import MLP
from preprocessing import DataPreprocessor

class Predictor:
    def __init__(self, model_path="model/model.pkl"):
        self.model = MLP.load(model_path)
        print(f"architecture: {self.model.layer_sizes}")

    def load(self, cls, file_name):
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / file_name
        d = pickle.loads(path.read_bytes())
        m = cls(hidden_layers=d["hidden_layers"], seed=d["seed"])
        m.layer_sizes = d["layer_sizes"]
        m.activations = d["activations"]
        m.weights = d["weights"]
        m.biases = d["biases"]
        return m

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true_onehot):
        proba = self.predict_proba(X)
        loss = self.model.categorical_crossentropy(y_true_onehot, proba)
        y_pred = self.predict(X)
        y_true = np.argmax(y_true_onehot, axis=1)
        my_accuracy = self._accuracy(y_true, y_pred)
        my_precision = self._precision(y_true, y_pred)
        my_recall = self._recall(y_true, y_pred)
        my_f1 = self._f1(y_true, y_pred)
        return loss, my_accuracy, my_precision, my_recall, my_f1

    def _accuracy(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return np.mean(y_true == y_pred)

    def _precision(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _recall(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _f1(self, y_true, y_pred):
        p = self._precision(y_true, y_pred)
        r = self._recall(y_true, y_pred)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0