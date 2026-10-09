import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

class MLP:
    def __init__(self, hidden_layers, epochs, batch_size, lr, seed):
        self.hidden_layers = list(hidden_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self.weights = []
        self.biases  = []
        self.activations = []
        self.layer_sizes = []

        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc":  [], "val_acc":  []}

    def relu(self, x): return np.maximum(0, x)
    def relu_d(self, x): return (x > 0).astype(x.dtype)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_d(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    def get_act(self, name):
        if name == "relu": return self.relu,    self.relu_d
        if name == "softmax": return self.softmax, self.softmax_d
        raise ValueError(f"Unknown activation: {name}")

    def bce(self, y_true, y_pred, eps=1e-12):
        y_true = y_true.reshape(-1)
        y_pred = np.clip(y_pred.reshape(-1), eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def categorical_crossentropy(self, y_true, y_pred, eps=1e-12):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def init_weights(self, layer_sizes, activations):
        rng = np.random.default_rng(self.seed)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights, self.biases = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in) if activations[i] == "relu" \
                    else np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        a = X
        cache = {"a": [a], "z": []}
        for i, name in enumerate(self.activations):
            fn, _ = self.get_act(name)
            z = a @ self.weights[i] + self.biases[i]
            a = fn(z)
            cache["z"].append(z)
            cache["a"].append(a)
        return a, cache

    def backward(self, cache, y_true):
        n = y_true.shape[0]
        y_pred = cache["a"][-1]
        gw = [None] * len(self.weights)
        gb = [None] * len(self.biases)

        delta = (y_pred - y_true) / n

        gw[-1] = cache["a"][-2].T @ delta
        gb[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            _, dfn = self.get_act(self.activations[i])
            delta = (delta @ self.weights[i + 1].T) * dfn(cache["z"][i])
            gw[i] = cache["a"][i].T @ delta
            gb[i] = np.sum(delta, axis=0, keepdims=True)
        return gw, gb

    def step(self, gw, gb):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * gw[i]
            self.biases[i] -= self.lr * gb[i]

    def run(self, X_train, y_train, X_val=None, y_val=None,
            model_path="model/model.pkl", plots_dir="results/plots"):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
        layer_sizes = [X_train.shape[1]] + self.hidden_layers + [2]
        activations = ["relu"] * len(self.hidden_layers) + ["softmax"]
        self.init_weights(layer_sizes, activations)
        print(f"architecture: {layer_sizes}, activations: {activations}")

        n = X_train.shape[0]
        rng = np.random.default_rng(self.seed)

        for epoch in range(1, self.epochs + 1):
            idx = rng.permutation(n)
            X_sh, y_sh = X_train[idx], y_train[idx]

            for i in range(0, n, self.batch_size):
                Xb = X_sh[i:i + self.batch_size]
                yb = y_sh[i:i + self.batch_size]
                _, cache = self.forward(Xb)
                gw, gb = self.backward(cache, yb)
                self.step(gw, gb)

            tp = self.forward(X_train)[0]
            tl = self.categorical_crossentropy(y_train, tp)
            ta = np.mean(np.argmax(tp, axis=1) == np.argmax(y_train, axis=1))
            if X_val is not None:
                vp = self.forward(X_val)[0]
                vl = self.categorical_crossentropy(y_val, vp)
                va = np.mean(np.argmax(vp, axis=1) == np.argmax(y_val, axis=1))
            else:
                vl, va = float("nan"), float("nan")

            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)
            self.history["train_acc"].append(ta)
            self.history["val_acc"].append(va)

            print(f"epoch {epoch:3d}/{self.epochs} | "
                  f"train_loss={tl:.4f} val_loss={vl:.4f} | "
                  f"train_acc={ta:.4f} val_acc={va:.4f}")

        self.save(model_path)
        self.plot(plots_dir)
        return self

    def plot(self, plots_dir):
        plots_dir = Path(__file__).parent.parent / 'images';
        plots_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(self.history["train_loss"], label="train")
        plt.plot(self.history["val_loss"],   label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss")
        plt.legend(); plt.grid(True)
        plt.savefig(plots_dir / "loss_curve.png", dpi=120); plt.close()

        plt.figure()
        plt.plot(self.history["train_acc"], label="train")
        plt.plot(self.history["val_acc"],   label="val")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy")
        plt.legend(); plt.grid(True)
        plt.savefig(plots_dir / "accuracy_curve.png", dpi=120); plt.close()

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.forward(X)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({
            "hidden_layers": self.hidden_layers,
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
            "weights": self.weights,
            "biases": self.biases,
            "seed": self.seed,
        }))

    @classmethod
    def load(cls, path):
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / path
        d = pickle.loads(path.read_bytes())
        m = cls(
            hidden_layers=d["hidden_layers"],
            epochs=100,
            batch_size=32,
            lr=0.01,
            seed=d["seed"],
        )
        m.layer_sizes = d["layer_sizes"]
        m.activations = d["activations"]
        m.weights = d["weights"]
        m.biases = d["biases"]
        return m