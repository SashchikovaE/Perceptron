from preprocessing import DataPreprocessor
from split import split_data
from train import MLP
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    split_data(
        input_file=f"data.csv",
        train_file=f"train.csv",
        test_file=f"test.csv"
    )
    train = DataPreprocessor()
    train.preprocess()

    test = DataPreprocessor('test.csv')
    test.preprocess(loaded=True)

    model = MLP(hidden_layers=(64, 32), epochs=100, batch_size=32, lr=0.001, seed=42)
    X_train = train.df.iloc[:, :-1].values
    y_train = train.df.iloc[:, -1].values
    X_val = test.df.iloc[:, :-1].values
    y_val = test.df.iloc[:, -1].values

    model.run(
        X_train, y_train,
        X_val, y_val,
        model_path="models/model.pkl",
        plots_dir="results/plots",
    )
