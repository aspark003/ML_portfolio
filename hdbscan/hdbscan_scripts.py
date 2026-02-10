import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
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

            self.scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

        except Exception as e:
            raise RuntimeError(f'invalid init: {e}')

    def b(self):
        try:
            #self.copy = self.copy.sample(frac=0.4, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)

            hdb = HDBSCAN(min_cluster_size=5, min_samples=10)

            label = hdb.fit_predict(x)

            for scores in self.scores:
                s = scores(x, label)
                print(f'{scores.__name__}:{s}')

            label_c, label_v = np.unique(label, return_counts=True)

            l_c = np.arange(len(label_c))

            plt.figure(figsize=(10,8))
            plt.scatter(l_c[label_c==-1], label_v[label_c==-1], c='red', label='Anomaly',s=50)
            plt.scatter(l_c[label_c!=-1], label_v[label_c!=-1], c='green', label='Cluster',s=30)
            plt.legend()
            plt.xlabel('CLUSTER INDEX')
            plt.ylabel('CLUSTER SIZE')
            plt.title('CLUSTER VS NOISE SIZE DIAGNOSTIC')
            plt.show()

            prob = hdb.probabilities_

            prob_label = np.arange(len(label))

            plt.figure(figsize=(10,8))
            plt.scatter(prob_label[label==-1], prob[label==-1], c='red', s=40, label='NOISE')
            plt.scatter(prob_label[label!=-1], prob[label!=-1], c='green',s=30, label='CLUSTER')
            plt.legend()
            plt.xlabel('DATA POINT INDEX')
            plt.ylabel('MEMBERSHIP PROBABILITIES')
            plt.title('CLUSTERS VS NOISE')
            plt.show()

            out_lier = hdb.outlier_scores_

            o_lier = np.arange(len(label))

            print(out_lier.shape)

            plt.figure(figsize=(10,8))
            plt.scatter(o_lier[label==-1], out_lier[label==-1], c='red', s=40, label='NOISE')
            plt.scatter(o_lier[label!=-1], out_lier[label!=-1], c='green', s=30, label='CLUSTER')
            plt.legend()
            plt.xlabel('DATA POINT INDEX')
            plt.ylabel('OUTLIER SCORE')
            plt.title('OUTLIER SCORES (HDBSCAN)')
            plt.show()

            prob_out = np.array(len(label))

            order = np.argsort(prob)

            plt.figure(figsize=(10,8))
            plt.plot(prob[order], label="Probability")
            plt.plot(out_lier[order],label="Outlier score")
            plt.legend()
            plt.xlabel('DATA INDEX')
            plt.ylabel('DATA VALUES')
            plt.title('PROBABILITIES AND OUTLIER SCORES (SORTED ARGUMENT')
            plt.show()

            cluster_per = hdb.cluster_persistence_

            cluster_len = np.arange(len(cluster_per))

            plt.figure(figsize=(10,8))
            plt.scatter(cluster_len, cluster_per, c='red',s=40)
            plt.title('HDBSCAN CLUSTER PERSISTENCE STABILITY')
            plt.xlabel('CLUSTER RANK (BY PERSISTENCE')
            plt.ylabel('CLUSTER PERSISTENCE')
            plt.show()

            prob_signal = pd.DataFrame({'probabilities': pd.Series(prob),
                                        'outlier scores': pd.Series(out_lier)})

            print(prob_signal.describe())

            print(pd.Series(cluster_per).describe())


        except Exception as e:
            raise RuntimeError(f'invalid hdbscan: {e}')

if __name__ == "__main__":
    abcd = ABCDE()
    abcd.b()
