import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class A:
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
            self.obj = self.copy.select_dtypes(include=['object']).columns

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

            self.scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

            #k = pd.api.types.is_numeric_dtype(self.copy['person_income'])

        except Exception as e:
            raise RuntimeError(f'invalid init:{e}')

    def b(self):
        try:
            x = self.preprocessor.fit_transform(self.copy)
            x_df = pd.DataFrame(x, columns=self.preprocessor.get_feature_names_out())

            db_scan = DBSCAN(eps=0.5, min_samples=6, metric='minkowski', p=2, algorithm='auto', n_jobs=-1)

            db_scan.fit(x_df)

            label = pd.Series(db_scan.labels_)

            plt.figure(figsize=(10, 8))
            plt.scatter(label[label == -1], label[label == -1], c='red', s=20, label='Noise')
            plt.scatter(label[label != -1], label[label != -1], c='green', s=10, label='Normal')
            plt.legend()
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.title('LABEL')
            plt.show()

            points = pd.Series(label).value_counts()

            plt.figure(figsize=(10, 8))
            plt.scatter(points.index[points.index == -1], points[points.index == -1], c='red', s=30, label='Noise')
            plt.scatter(points.index[points.index != -1], points[points.index != -1], c='green', s=10, label='Normal')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.legend()
            plt.title('POINTS')
            plt.show()

            cluster = pd.Series(points).value_counts()

            c_index = pd.Series(points).index.to_numpy()
            c_value = pd.Series(points).to_numpy()

            plt.figure(figsize=(10,8))
            plt.scatter(np.arange(len(c_index[c_index == -1])), c_value[c_index==-1], c='red', s=20, label='Noise')
            plt.scatter(np.arange(len(c_index[c_index != -1])), c_value[c_index !=-1], c='green',s=10, label='Normal')
            plt.legend()
            plt.xlabel('INDEX')
            plt.ylabel('VALUE')
            plt.title('CLUSTER')
            plt.show()

            for scores in self.scores:
                s = scores(x_df, label)
                print(f'{scores.__name__}: {s}')

        except Exception as e:
            raise RuntimeError(f'invalid dbscan: {e}')


if __name__ == "__main__":
    a = A()
    a.b()
