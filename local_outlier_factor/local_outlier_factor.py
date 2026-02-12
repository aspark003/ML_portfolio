import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class I:
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
            #self.copy = self.copy.sample(frac=0.2, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)
            lof = LocalOutlierFactor(n_neighbors=5, contamination=0.10, metric='minkowski', p=1, leaf_size=30)

            lof.fit(x)

            nof = lof.negative_outlier_factor_

            l_x, l_y = np.unique(nof, return_counts=True)

            plt.figure(figsize=(10,8))
            plt.scatter(l_x, l_y, s=50, c='red')
            plt.title('LOCAL OUTLIER FACTOR (NEGATIVE OUTLIER FACTOR)')
            plt.xlabel('FREQUENCY (COUNT OF SCORES)')
            plt.ylabel('SCORE VS FREQUENCY')
            plt.show()

            print(pd.DataFrame({'label': l_x,
                                'value': l_y}).describe())

            off_set = lof.offset_

            o_score, o_count = np.unique(off_set, return_counts=True)

            plt.figure(figsize=(10,8))
            plt.scatter(np.argsort(l_x[l_x > o_score]), l_x[l_x > o_score], c='red', marker='x')
            plt.scatter(np.argsort(l_x[l_x < o_score]), l_x[l_x < o_score], c='green', marker='o')
            plt.title('LOCAL OUTLIER FACTOR CLASSIFICATION')
            plt.xlabel('SAMPLE INDEX')
            plt.ylabel('NEGATIVE OUTLIER FACTOR (SCORES)')
            plt.show()


        except Exception as e:
            raise RuntimeError(f'invalid local outlier factor: {e}')


if __name__ == "__main__":
    abc = I()
    abc.b()
