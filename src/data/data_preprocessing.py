import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    def load_data(self, file_name: str) -> pd.DataFrame:
        """Load dataset from a given file path with exception handling."""
        file_path = os.path.join(self.input_path, file_name)
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"Error loading data from {file_path}: {e}")  
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the column mean."""
        try:
            for col in df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())  
            return df 
        except Exception as e:
            raise Exception(f"Error filling missing values: {e}")
    
    def scale_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize features by removing the mean and scaling to unit variance."""
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df.iloc[:, :-1]) 
            scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])  
            scaled_df[df.columns[-1]] = df.iloc[:, -1].values  
            return scaled_df
        except Exception as e:
            raise Exception(f"Error scaling data: {e}")
    
    def save_data(self, df: pd.DataFrame, file_name: str) -> None:
        """Save the processed data to a CSV file."""
        try:
            file_path = os.path.join(self.output_path, file_name)
            df.to_csv(file_path, index=False)
            print(f"Processed file saved: {file_path}")
        except Exception as e:
            raise Exception(f"Error saving data to {file_path}: {e}")
    
    def execute(self):
        """Execute the data preprocessing pipeline."""
        try:
            train_data = self.load_data("train_data.csv")
            test_data = self.load_data("test_data.csv")

            train_data = self.fill_missing_values(train_data)
            test_data = self.fill_missing_values(test_data)

            train_data = self.scale_data(train_data)
            test_data = self.scale_data(test_data)

            self.save_data(train_data, "train_processed.csv")
            self.save_data(test_data, "test_processed.csv")
            print("Data preprocessing complete.")
        except Exception as e:
            print(f"Error during data preprocessing: {e}")

if __name__ == "__main__":
    input_path = "data/raw"
    output_path = "data/processed"
    preprocessing = DataPreprocessing(input_path, output_path)
    preprocessing.execute()



# train_data = pd.read_csv("./data/raw/train_data.csv")
# test_data = pd.read_csv("./data/raw/test_data.csv")

# train_processed_data = fill_missing_value(train_data)
# train_processed_data = scale_data(train_processed_data)

# test_processed_data = fill_missing_value(test_data)
# test_processed_data = scale_data(test_processed_data)

# data_path = os.path.join("data", "processed")
# os.makedirs(data_path, exist_ok=True)

# train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
# test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

# print("Data preprocessing complete. Processed files saved in 'data/processed'.")
