import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import yaml

class ModelBuilding:
    def __init__(self, config_file: str):
        """Initialize Model Building with config file."""
        self.config_file = config_file
        self.config = self.load_params()
        self.model_dir = os.path.join("models")
        os.makedirs(self.model_dir, exist_ok=True)

    def load_params(self):
        """Load model hyperparameters from a YAML file."""
        try:
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Error loading parameters from {self.config_file}: {e}")

    def load_data(self, file_path: str):
        """Load dataset from a CSV file."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {e}")

    def split_data(self, df: pd.DataFrame):
        """Split dataset into features (X) and target (y)."""
        try:
            X = df.drop("Potability", axis=1)
            y = df["Potability"].astype(int) 
            return X, y
        except Exception as e:
            raise Exception(f"Error splitting data: {e}")

    def train_model(self, X, y):
        """Train the RandomForest model with parameters from config."""
        try:
            n_estimators = self.config["model_building"]["n_estimators"]
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X, y)
            return model
        except Exception as e:
            raise Exception(f"Error training model: {e}")

    def save_model(self, model):
        """Save the trained model as a pickle file."""
        try:
            model_path = os.path.join(self.model_dir, "rf_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved successfully at {model_path}")
        except Exception as e:
            raise Exception(f"Error saving model: {e}")

    def execute(self, data_path):
        """Execute the model training pipeline."""
        try:
            df = self.load_data(data_path)
            X, y = self.split_data(df)
            model = self.train_model(X, y)
            self.save_model(model)
        except Exception as e:
            print(f"Error during model building: {e}")

if __name__ == "__main__":
    config_path = "params.yaml"
    data_path = "./data/processed/train_processed_with_median.csv"

    model_builder = ModelBuilding(config_path)
    model_builder.execute(data_path)






















# train_data=pd.read_csv('./data/processed/train_processed.csv')

# X_train = train_data.drop("Potability",axis=1)
# y_train = train_data['Potability']

# n_estimators = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']

# rf =RandomForestClassifier(n_estimators=n_estimators)
# rf.fit(X_train,y_train)

# with open('rf_model.pkl',"wb") as f:
#     pickle.dump(rf,f)

# print("Model building complete and saved in 'rf_model.pkl'.")
