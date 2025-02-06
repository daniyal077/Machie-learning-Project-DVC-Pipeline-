import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.scaler = StandardScaler()  

    def load_data(self, file_name: str) -> pd.DataFrame:
        """Load dataset from a given file path with exception handling."""
        file_path = os.path.join(self.input_path, file_name)
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")  

    def fill_missing_values_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the column mean for numerical columns."""
        try:
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].mean())  
            return df 
        except Exception as e:
            raise ValueError(f"Error filling missing values: {e}")
    
    def scale_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Standardize numerical features in training and test set using StandardScaler."""
        try:
            num_cols = train_df.select_dtypes(include=[np.number]).columns  

            
            self.scaler.fit(train_df[num_cols])

            train_df[num_cols] = self.scaler.transform(train_df[num_cols])
            test_df[num_cols] = self.scaler.transform(test_df[num_cols])

            return train_df, test_df
        except Exception as e:
            raise ValueError(f"Error scaling data: {e}")
    
    def save_data(self, df: pd.DataFrame, file_name: str) -> None:
        """Save the processed data to a CSV file."""
        try:
            file_path = os.path.join(self.output_path, file_name)
            df.to_csv(file_path, index=False)
            print(f"Processed file saved: {file_path}")
        except Exception as e:
            raise ValueError(f"Error saving data to {file_path}: {e}")
    
    def execute(self):
        """Execute the data preprocessing pipeline."""
        try:
            train_data = self.load_data("train_data.csv")
            test_data = self.load_data("test_data.csv")

            train_data = self.fill_missing_values_mean(train_data)
            test_data = self.fill_missing_values_mean(test_data)

            train_data, test_data = self.scale_data(train_data, test_data)

            self.save_data(train_data, "train_processed_with_mean.csv")
            self.save_data(test_data, "test_processed_with_mean.csv")

            print("Data preprocessing complete.")
        except Exception as e:
            print(f"Error during data preprocessing: {e}")

if __name__ == "__main__":
    input_path = "data/raw"
    output_path = "data/processed"
    preprocessing = DataPreprocessing(input_path, output_path)
    preprocessing.execute()
