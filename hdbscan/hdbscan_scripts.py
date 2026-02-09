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
            self.copy = self.copy.sample(frac=0.2, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)

            hdb = HDBSCAN(min_cluster_size=6, min_samples=5, metric='minkowski', p=2,
                          cluster_selection_method='eom', prediction_data=True)

            hdb.fit(x)
            label = hdb.labels_

            for score_func in self.scores:
                s = score_func(x, label)
                print(f'{score_func.__name__}: {s}')

            plt.figure(figsize=(10,8))
            plt.scatter(x[label==-1,0], x[label==-1,1], c='red', s=50, label='Noise')
            plt.scatter(x[label!=-1,0], x[label!=-1,1], c='green', s=20, label='Clustered')
            plt.legend()
            plt.xlabel('Feature 0')
            plt.ylabel('Feature 1')
            plt.title('HDBSCAN - First two dimensions (limited view)')
            plt.show()

            condensed_tree = hdb.condensed_tree_

            condensed_df = condensed_tree.to_pandas()

            print(condensed_df.describe())


            condensed_tree.plot()
            plt.title('Condensed Tree - Full Hierarchy')
            plt.show()

            condensed_tree.plot(select_clusters=True, selection_palette='dark')
            plt.title('Condensed Tree - Selected Clusters Highlighted')
            plt.show()

            prob = hdb.probabilities_
            out = hdb.outlier_scores_

            approx_labels, approx_probs = approximate_predict(hdb, x)

            approx_outlier = approx_labels[label==-1]
            approx_prob_outlier = approx_probs[label==-1]


            hd_signal = pd.DataFrame({
                'label': label,
                'probability': prob,
                'outlier_score': out,
                'approx_label': approx_labels,
                'approx_prob': approx_probs
            })


            approx_signal = pd.DataFrame({'approximate labels': approx_outlier,
                                          'approximate prob': approx_prob_outlier})

            print(approx_signal.describe().to_string())


        except Exception as e:
            raise RuntimeError(f'invalid hdbscan: {e}')

if __name__ == "__main__":
    abcd = ABCDE()
    abcd.b()
