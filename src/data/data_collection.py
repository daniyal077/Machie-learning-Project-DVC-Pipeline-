import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.data_path = os.path.join('data', 'raw')
        os.makedirs(self.data_path, exist_ok=True)
    
    def load_config(self):
        """Load configuration from YAML file with exception handling."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Error loading parameters from {self.config_path}: {e}")
    
    def load_data(self, url: str):
        """Load dataset from a given URL with exception handling."""
        try:
            return pd.read_csv(url)
        except Exception as e:
            raise Exception(f"Error loading data from {url}: {e}")
    
    def split_data(self, df: pd.DataFrame):
        """Split the dataset into train and test sets with exception handling."""
        try:
            test_size = self.config['data_collection']['test_size']
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            return train_data, test_data
        except ValueError as e:
            raise Exception(f"Error splitting data: {e}")
    
    def save_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Save train and test datasets to CSV files with exception handling."""
        try:
            train_data.to_csv(os.path.join(self.data_path, 'train_data.csv'), index=False)
            test_data.to_csv(os.path.join(self.data_path, 'test_data.csv'), index=False)
            print("Data collection complete. Processed files saved in 'data/raw'.")
        except Exception as e:
            raise Exception(f"Error saving data to filepath {self.data_path}: {e}")
    
    def execute(self, data_url: str):
        """Execute the entire data ingestion pipeline with exception handling."""
        try:
            df = self.load_data(data_url)
            train_data, test_data = self.split_data(df)
            self.save_data(train_data, test_data)
        except Exception as e:
            print(f"Error during data ingestion process: {e}")

if __name__ == "__main__":
    config_path = 'params.yaml'
    data_url = "https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv"
    ingestion = DataIngestion(config_path)
    ingestion.execute(data_url)












# ---------------------------------------------------------------------------------------
# df = pd.read_csv("https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv")

# test_size = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']

# train_data,test_data=train_test_split(df,test_size=test_size,random_state=42)

# data_path=os.path.join('data','raw')
# os.makedirs(data_path)

# train_data.to_csv(os.path.join(data_path,'train_data.csv'),index=False)
# test_data.to_csv(os.path.join(data_path,'test_data.csv'),index=False)

# print("Data collection complete. Processed files saved in 'data/raw'.")

