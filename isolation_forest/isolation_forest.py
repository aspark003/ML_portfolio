import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class ABCDEF:
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

            self.n_simple = SimpleImputer(strategy='median', add_indicator=True)
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
            iso = IsolationForest(n_estimators=200, max_samples=10, contamination=0.10, random_state=42, n_jobs=-1)

            label = iso.fit_predict(x)

            decision = iso.decision_function(x)

            plt.figure(figsize=(10,8))
            plt.scatter(np.sort(label), np.sort(decision), c='red')
            plt.title('ISOLATION FOREST(DECISION FUNCTION SCORES')
            plt.xlabel('SAMPLE INDEX')
            plt.ylabel('DECISION FUNCTION SCORES')
            plt.show()

            print(pd.DataFrame({'sample index': label,
                                'decision function scores': decision}).describe())

            d, i = np.unique(decision, return_counts=True)

            plt.figure(figsize=(10,8))
            plt.scatter(np.sort(d), np.sort(i), c='red')
            plt.xlabel('UNIQUE VALUES')
            plt.ylabel('COUNT OBSERVATION')
            plt.title('ISOLATION FOREST - SCORE FREQUENCY DISTRIBUTION')
            plt.show()

            print(pd.DataFrame({'unique values': d,
                                'count observation': i}).describe())

            #print(self.copy.head(1).to_string())
        except Exception as e:
            raise RuntimeError(f'invalid isolation forest: {e}')


if __name__ == "__main__":
    abcdef = ABCDEF()
    abcdef.b()
