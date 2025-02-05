import pandas as pd
import pickle
import numpy as np
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


import pandas as pd
import pickle
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, model_path: str, test_data_path: str, result_path: str = "results"):
        """Initialize Model Evaluation with model file and test data path."""
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)  
    def load_data(self):
        """Load test dataset from CSV file."""
        try:
            return pd.read_csv(self.test_data_path)
        except Exception as e:
            raise Exception(f"Error loading test data from {self.test_data_path}: {e}")

    def load_model(self):
        """Load trained model from a pickle file."""
        try:
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model from {self.model_path}: {e}")

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return metrics."""
        try:
            y_pred = model.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred)
            }
            return metrics
        except Exception as e:
            raise Exception(f"Error during model evaluation: {e}")

    def save_results(self, metrics):
        """Save evaluation metrics to a JSON file."""
        try:
            result_file = os.path.join(self.result_path, "result_metrics.json")
            with open(result_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Evaluation metrics saved successfully at {result_file}")
        except Exception as e:
            raise Exception(f"Error saving evaluation metrics: {e}")

    def execute(self):
        """Execute the entire model evaluation pipeline."""
        try:
            test_data = self.load_data()
            model = self.load_model()

            X_test = test_data.drop("Potability", axis=1)
            y_test = test_data["Potability"]

            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Print metrics
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            self.save_results(metrics)
        except Exception as e:
            print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    model_path="./models/rf_model.pkl"
    test_data_path="./data/processed/test_processed.csv"
    model_eval = ModelEvaluation(model_path, test_data_path)
    model_eval.execute()


























# test_data = pd.read_csv('./data/processed/test_processed.csv')

# with open('rf_model.pkl','rb') as f:
#     model = pickle.load(f)


# X_test = test_data.drop('Potability',axis=1)
# y_test = test_data['Potability']

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")


# result_dict = {
#     'Accuracy':accuracy,
#     'Precision':precision,
#     'Recall':recall,
#     'F1 Score':f1
# }

# with open("result_metrics.json",'w') as f:
#     json.dump(result_dict,f,indent=4)


# print("Model evaluation complete and saved result in 'result_metrics.json'.")

