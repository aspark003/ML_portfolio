import pandas as pd
import numpy as np
import os


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

        if extension == ".csv":
            self.df = pd.read_csv(self.file)
            print("CSV file loaded successfully.")

        elif extension == '.xlsx':
            self.df = pd.read_excel(self.file, engine='openpyxl')
            print('EXCEL file loaded successfully')

        print(self.df.head().to_string())

        self.df.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_load_file.csv', index=False)


if __name__ == "__main__":
    gr = FileLoader('c:/Users/anton/OneDrive/gov_finance/soft_gl_auto.xlsx')
    gr.load()
