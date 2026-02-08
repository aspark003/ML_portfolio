import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN, approximate_predict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ABCDE:
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

            self.scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

            #k = pd.api.types.is_numeric_dtype(self.copy['person_income'])

        except Exception as e:
            raise RuntimeError(f'invalid init:{e}')

    def b(self):
        try:

            self.copy = self.copy.sample(frac=0.2, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)

            hdb = HDBSCAN(min_cluster_size=6,min_samples=3,metric="minkowski",p=2)

            hdb.fit(x)
            label = hdb.labels_

            la = pd.Series(label).to_numpy()

            plt.figure(figsize=(10, 8))
            plt.scatter(la[la == -1], la[la == -1], c='red', s=40, label='Noise', marker='x')
            plt.scatter(la[la != -1], la[la != -1], c='green', s=30, label='Cluster', marker='o')
            plt.title('HDBSCAN Cluster Labels (Noise vs Cluster)')
            plt.legend()
            plt.xlabel('Cluster Label')
            plt.ylabel('Cluster Label')
            plt.show()

            points = pd.Series(label).value_counts()

            p_index = points.index.to_numpy()
            p_value = points.to_numpy()

            p_len = np.arange(len(p_index))

            plt.figure(figsize=(10, 8))
            plt.scatter(p_len[p_index == -1], p_value[p_index == -1], c='red', s=40, label='Noise')
            plt.scatter(p_len[p_index != -1], p_value[p_index != -1], c='green', s=30, label='Cluster')
            plt.legend()
            plt.xlabel('Label Index')
            plt.ylabel('Number of Points')
            plt.title('HDBSCAN Cluster Sizes (Noise vs Cluster)')
            plt.show()

            cluster = pd.Series(points).value_counts()
            c_index = pd.Series(cluster).index.to_numpy()
            c_value = pd.Series(cluster).to_numpy()

            plt.figure()
            plt.scatter(np.arange(len(c_index[c_index == -1])), c_value[c_index == -1], c='red', label='Noise', s=40)
            plt.scatter(np.arange(len(c_index[c_index != -1])), c_value[c_index != -1], c='green', s=30,
                        label='Cluster')
            plt.title('HDBSCAN Cluster Counts (Noise vs Cluster)')
            plt.xlabel('Label Index')
            plt.ylabel('Number of Points')
            plt.legend()
            plt.show()

            lpc = pd.DataFrame({'label': pd.Series(label),
                                'points': pd.Series(points),
                                'cluster': pd.Series(cluster)})

            print(lpc.describe())

            probabilities = hdb.probabilities_

            indices = np.arange(len(probabilities))

            plt.figure(figsize=(10, 8))
            plt.scatter(indices[la == -1], probabilities[la == -1], c='red', s=40, label='Noise', marker='x')
            plt.scatter(indices[la != -1], probabilities[la != -1], c='green', s=30, label='Cluster', marker='o')
            plt.title('HDBSCAN Membership Probabilities')
            plt.xlabel('INDEX')
            plt.ylabel('Membership Probability')
            plt.legend()
            plt.show()

            outlier_scores = hdb.outlier_scores_

            plt.figure(figsize=(10, 8))
            plt.scatter(indices[la == -1], outlier_scores[la == -1], c='red', s=40, label='Noise', marker='x')
            plt.scatter(indices[la != -1], outlier_scores[la != -1], c='green', s=30, label='Cluster', marker='o')
            plt.title('HDBSCAN Outlier Scores')
            plt.xlabel('INDEX')
            plt.ylabel('Outlier Score')
            plt.legend()
            plt.show()

            po_df = pd.DataFrame({'probability': probabilities,
                                  'indices': indices})

            print(po_df.describe())


        except Exception as e:
            raise RuntimeError(f'invalid hdbscan: {e}')




if __name__ == "__main__":
    abcd = ABCDE()
    abcd.b()
