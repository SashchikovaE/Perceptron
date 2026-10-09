from preprocessing import DataPreprocessor
from split import split_data
from train import MLP
from pathlib import Path
from predict import Predictor
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    split_data(
        input_file=f"data.csv",
        train_file=f"train.csv",
        val_file=f"val.csv",
        test_file=f"test.csv"
    )
    train = DataPreprocessor()
    drop_cols = train.preprocess()

    val = DataPreprocessor('val.csv')
    val.preprocess(selection='val', drop_cols=drop_cols)

    test = DataPreprocessor('test.csv')
    test.preprocess(loaded=True, selection='test', drop_cols=drop_cols)

    model = MLP(hidden_layers=(64, 32), epochs=500, batch_size=32, lr=0.001, seed=42)
    X_train = train.df.iloc[:, :-1].values
    y_train = train.df.iloc[:, -1].values
    X_val = val.df.iloc[:, :-1].values
    y_val = val.df.iloc[:, -1].values

    y_train_1hot = np.eye(2)[y_train]
    y_val_1hot = np.eye(2)[y_val]

    model.run(
        X_train, y_train_1hot,
        X_val, y_val_1hot,
        model_path="model/model.pkl",
        plots_dir="results/plots",
    )

    X_test = test.df.iloc[:, :-1].values
    y_test = test.df.iloc[:, -1].values.astype(int)
    y_test_onehot = np.eye(2)[y_test]

    predictor = Predictor("model/model.pkl")
    loss, accuracy, precision, recall, f1 = predictor.evaluate(X_test, y_test_onehot)

    print(f"accuracy: {accuracy:.4f}")
    print(f"precision: {accuracy:.4f}")
    print(f"recall: {accuracy:.4f}")
    print(f"f1: {accuracy:.4f}")
    print(f"test loss (CCE): {loss:.4f}")

