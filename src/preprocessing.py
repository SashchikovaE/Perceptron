import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

class DataPreprocessor:
    def __init__(self, file='train.csv'):
        data_path = Path(__file__).parent.parent / 'data' / file
        self.df = pd.read_csv(str(data_path))

    def make_header(self):
        self.df.columns = [str(i) for i in range(1, len(self.df.columns) + 1)]

    def encode_categorical_data(self, col, mapping):
        self.df[col] = self.df[col].map(mapping)

    def analyze_empty_rows_and_cols(self):
        pd.set_option('display.width', None)
        print(self.df[self.df.isna().any(axis=1)])
        print(self.df['2'].unique())
        print(self.df.isnull().sum())

    def visualize_target_distribution(self):
        sns.countplot(data=self.df, x='2')
        plt.title('Target distribution')
        file = Path(__file__).parent.parent / 'images/target_distribution.png'
        file.parent.mkdir(exist_ok=True)
        plt.savefig(file)
        plt.show()

    def visualize_correlation_matrix(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(28.8, 19.2))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f")
        plt.title('Correlation matrix')
        file = Path(__file__).parent.parent / 'images/correlation_matrix.png'
        file.parent.mkdir(exist_ok=True)
        plt.savefig(file, bbox_inches='tight')
        plt.show()

    def get_path(self):
        cur_file = Path(__file__)
        model_dir = cur_file.parent.parent / 'api_files'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'normalization.joblib'
        return model_path

    def normalize_data(self, loaded=False):
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        non_binary_cols = [
            col for col in numeric_cols if self.df[col].nunique() > 2]
        if not loaded:
            mean = self.df[non_binary_cols].mean(axis=0)
            std = self.df[non_binary_cols].std(axis=0)
            params = {
                'mean': mean,
                'std': std,
                'non_binary_cols': non_binary_cols
            }
            model_path = self.get_path()
            joblib.dump(params, model_path)
        else:
            model_path = self.get_path()
            params = joblib.load(model_path)
            mean = params['mean']
            std = params['std']
            non_binary_cols = params['non_binary_cols']
        self.df[non_binary_cols] = (
            self.df[non_binary_cols] - mean) / (std + 1e-10)
        return self.df

    def print_table(self):
        pd.set_option('display.width', None)
        print(self.df)

    def move_target_to_end(self):
        if '2' in self.df.columns:
            target = self.df['2']
            self.df = self.df.drop('2', axis=1)
            self.df['target'] = target

    def preprocess(self, loaded=False):
        self.make_header()
        self.analyze_empty_rows_and_cols()
        self.encode_categorical_data('2', {'M': 0, 'B': 1})
        print(self.df.describe())
        self.visualize_target_distribution()
        #self.visualize_correlation_matrix()
        self.normalize_data(loaded=False)
        self.move_target_to_end()
        self.print_table()
