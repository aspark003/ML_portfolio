import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import os
from zipfile import BadZipFile, ZipFile
import openpyxl

class FileLoader:
    def __init__(self, file=None):
        self.file = file
        self.df = None

    def load(self, file=None):
        # Update file path if a new one is provided
        if file:
            self.file = file

        if not self.file:
            raise ValueError("File path not available")

        # Get file extension
        extension = os.path.splitext(self.file)[1].lower()

        try:
   
            if extension == ".xlsx":
                try:
                    # Ensure the file is a proper zip (xlsx)
                    with ZipFile(self.file, 'r') as zip_ref:
                        corrupt_file = zip_ref.testzip()
                        if corrupt_file:
                            raise BadZipFile(f"Corrupt file inside archive: {corrupt_file}")
                    # File is valid Excel, load it
                    self.df = pd.read_excel(self.file, engine='openpyxl')
                    print("Excel file loaded successfully.")

                except BadZipFile:
                    raise ValueError(f"Excel file is corrupted or not a valid .xlsx: {self.file}")

            elif extension == ".csv":
                self.df = pd.read_csv(self.file)
                print("CSV file loaded successfully.")

            elif extension == ".tsv":
                self.df = pd.read_csv(self.file, sep="\t")
                print("TSV file loaded successfully.")

            elif extension in [".json", ".jsonl"]:
                try:
                    if extension == ".jsonl":
                        self.df = pd.read_json(self.file, lines=True)
                    else:
                        self.df = pd.read_json(self.file)
                    print("JSON file loaded successfully.")
                except ValueError as e:
                    raise ValueError(f"Invalid JSON file: {e}")

            elif extension in [".txt", ".text"]:
                with open(self.file, "r", encoding="utf-8", errors="ignore") as f:
                    self.df = pd.DataFrame([line.strip() for line in f.readlines()], columns=["text"])
                print("Text file loaded successfully.")

            else:
                raise ValueError(f"Unsupported file type: {extension}")

        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")

        if self.df is not None:
            print(self.df.head().to_string())
        self.df.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto1.csv', index=False)
        return self.df



if __name__ == "__main__":
    fc = FileLoader('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto.xlsx')
    df = fc.load()
