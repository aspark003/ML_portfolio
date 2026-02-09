import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class ABC:
    def __init__(self):
        try:
            self.df = pd.read_csv('c:/Users/anton/risk/credit.csv', encoding='utf-8-sig', engine='python')
            self.copy = self.df.copy()

            self.copy['person_age'] = self.copy['person_age'].astype(int)
            self.copy['person_income'] = self.copy['person_income'].astype(float)
            self.copy['person_home_ownership'] = self.copy['person_home_ownership'].astype(object)
            self.copy['person_emp_length'] = self.copy['person_emp_length'].astype(float)
            self.copy['loan_intent'] = self.copy['loan_intent'].astype(object)
            self.copy['loan_grade'] = self.copy['loan_grade'].astype(object)
            self.copy['loan_amnt'] = self.copy['loan_amnt'].astype(float)
            self.copy['loan_int_rate'] = self.copy['loan_int_rate'].astype(float)
            self.copy['loan_percent_income'] = self.copy['loan_percent_income'].astype(float)
            self.copy['cb_person_cred_hist_length'] = self.copy['cb_person_cred_hist_length'].astype(int)

            self.copy = self.copy.drop(columns=['person_age', 'person_income','person_emp_length','person_home_ownership', 'loan_grade', 'loan_status','cb_person_default_on_file'])


            self.num = self.copy.select_dtypes(include=['number']).columns
            self.obj = self.copy.select_dtypes(include=['object', 'string']).columns

            self.mm = MinMaxScaler()
            self.ohe = OneHotEncoder(drop=None, handle_unknown='ignore', sparse_output=False)

            self.n_simple = SimpleImputer(strategy='mean', add_indicator=True)
            self.o_simple = SimpleImputer(strategy='constant', fill_value='missing')

            self.n_pipe = Pipeline([('n_impute', self.n_simple),
                                    ('num', self.mm)])

            self.o_pipe = Pipeline([('o_impute', self.o_simple),
                                    ('obj', self.ohe)])

            self.preprocessor = ColumnTransformer([('scaler', self.n_pipe, self.num),
                                                   ('encoder', self.o_pipe, self.obj)])

            #k = pd.api.types.is_numeric_dtype(self.copy['person_income'])

        except Exception as e:
            raise RuntimeError(f'invalid init:{e}')

    def b(self):
        try:
            x = self.preprocessor.fit_transform(self.copy)
            nn = NearestNeighbors(n_neighbors=6, metric='minkowski', p=1,algorithm='auto', leaf_size=30, n_jobs=-1)
            nn.fit(x)

            distance, indices = nn.kneighbors(x)

            plt.figure(figsize=(10,8))
            plt.scatter(distance[:,0], distance[:,1], c='red',s=20, label='0-1')
            plt.scatter(distance[:,1], distance[:,2], c='green',s=20, label='1-2')
            plt.scatter(distance[:,2], distance[:,3], c='blue', s=10, label='2-3')
            plt.scatter(distance[:,3], distance[:,4], c='yellow',s=10, label='3-4')
            plt.title('NEAREST NEIGHBOR DISTANCE MEASURES')
            plt.legend()
            plt.xlabel('INDEX')
            plt.ylabel('VALUE')
            plt.show()

            plt.figure(figsize=(10,8))
            plt.scatter(indices[:, 0], indices[:, 1], c='red', s= 20, label='0-1')
            plt.scatter(indices[:, 1], indices[:, 2], c='green', s=20, label='1-2')
            plt.scatter(indices[:, 2], indices[:, 3], c='blue', s=10, label='2-3')
            plt.scatter(indices[:, 3], indices[:, 4], c='yellow', s=10, label='3-4')
            plt.title('NEAREST NEIGHBOR INDICES MEASURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.legend()
            plt.show()

            d_df = pd.DataFrame(MinMaxScaler().fit_transform(distance), columns=[f"Distance: {k}" for k in range(distance.shape[1])])

            i_df = pd.DataFrame(MinMaxScaler().fit_transform(indices), columns=[f'Indices: {k}' for k in range(indices.shape[1])])

            print(d_df.describe().to_string())
            print()
            print(i_df.describe().to_string())

        except Exception as e:
            raise RuntimeError(f'invalid nearest neighbors: {e}')


if __name__ == "__main__":
    abc = ABC()
    abc.b()
