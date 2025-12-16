import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

class GovRegression:
    def __init__(self, file_path, header=2):
        try:
            self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python', header=header)
            self.df.insert(0, 'id', self.df.index+1)
            self.df.columns = self.df.columns.str.replace('-', ' ', regex=True).str.lower().str.strip()
            self.df.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto6.csv',index=False)
            #print(self.df.head().to_string())

            self.copy = self.df.copy()
            self.start = self.copy.copy()
            self.start = self.start.drop(columns=['task organization', 'bfy', 'ba bsa bli', 'fund', 'limit'])
            self.start.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto3.csv',index=False)
            print(self.start.head().to_string())
            print(self.start.shape)

            self.copy = self.copy.drop(columns=['task organization', 'bfy', 'ba bsa bli', 'fund', 'limit', 'project number', 'task number', 'expenditure type', 'class category', 'class code', 'budget authority', 'commitments', 'obligations', 'non labor expenditures'])
            #print(self.copy.head().to_string())
            #print(self.copy.dtypes)
            self.copy.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto2.csv',index=False)

        except Exception as e: print(f'invalid file: {e}')


if __name__ == "__main__":
    rc = GovRegression('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto1.csv')
