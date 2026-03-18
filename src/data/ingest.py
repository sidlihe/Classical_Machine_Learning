import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter,dataset_load
import os
filename = "creditcard_fraud.csv"

def load_kaggle_dataset(dataset: str, save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    
    # This downloads the entire dataset and returns the folder path
    path = kagglehub.dataset_download(dataset)
    print("Dataset downloaded to:", path)
    
    # Find the CSV inside (adjust pattern if needed)
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the dataset")
    
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    
    output_path = os.path.join(save_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    return df

if __name__ == "__main__":
    dataset_path = "mlg-ulb/creditcardfraud"
    df = load_kaggle_dataset(dataset_path)
    print("First 5 records:", df.head())
