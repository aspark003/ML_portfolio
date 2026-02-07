import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import OPTICS
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class ABCD:
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
            self.copy['loan_status'] = self.copy['loan_status'].astype(bool)
            self.copy['loan_percent_income'] = self.copy['loan_percent_income'].astype(float)
            self.copy['cb_person_default_on_file'] = self.copy['cb_person_default_on_file'].astype(bool)
            self.copy['cb_person_cred_hist_length'] = self.copy['cb_person_cred_hist_length'].astype(int)

            self.copy = self.copy.drop(columns=['person_age', 'person_income','person_emp_length','person_home_ownership', 'loan_grade'])


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
            #self.copy = self.copy.sample(frac=0.1, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)

            o = OPTICS(min_samples=5, metric='minkowski', p=2, algorithm='auto')

            o.fit(x)

            reach = o.reachability_

            r = np.isfinite(reach)

            r_max = np.isfinite(reach).max()

            reach_replace = np.where(r, reach, r_max)

            ordering = o.ordering_

            r_reach = pd.Series(reach_replace)

            o_order = pd.Series(ordering)

            ro = r_reach[o_order]

            plt.figure(figsize=(10, 8))
            plt.scatter(r_reach.index, r_reach[o_order.index], c='red', s=100, label='Ordering')
            plt.plot(r_reach[o_order.index], linestyle='--', marker='o', c='green', label='Reachability')
            plt.title('ORDERING / REACHABILITY')
            plt.legend()
            plt.xlabel('INDEX')
            plt.ylabel('VALUE')
            plt.show()

            o_df = pd.DataFrame({'reachability': pd.Series(MinMaxScaler().fit_transform(reach_replace.reshape(-1,1)).ravel()),
                                 'ordering': pd.Series(MinMaxScaler().fit_transform(ordering.reshape(-1,1)).ravel())})

            print(o_df.describe())


        except Exception as e:
            raise RuntimeError(f'invalid nearest neighbor: {e}')




if __name__ == "__main__":
    abcd = ABCD()
    abcd.b()
