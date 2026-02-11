import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

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
            self.copy = self.copy.sample(frac=0.2, random_state=42)
            x = self.preprocessor.fit_transform(self.copy)
            gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)

            label = gmm.fit_predict(x)

            score_sample = gmm.score_samples(x)

            score_l, score_v = np.unique(score_sample, return_counts=True)

            plt.figure(figsize=(10,8))
            plt.scatter(score_l, score_v)
            plt.xlabel('LOG DENSITY (SCORE SAMPLE)')
            plt.ylabel('FREQUENCY (COUNT OF IDENTICAL SCORES')
            plt.title('GAUSSIAN MIXTURE LOG DENSITY DISTRIBUTION')
            plt.show()

            print(pd.DataFrame({'score sample index': score_l,
                                'score sample counts': score_v}).describe())

            prob = gmm.predict_proba(x)

            prob_index, prob_value = np.unique(prob, return_counts=True)

            plt.figure(figsize=(10,8))
            sns.kdeplot(score_sample)
            plt.xlabel('LOG DENSITY (SCORE SAMPLES)')
            plt.ylabel('DENSITY')
            plt.title('GAUSSIAN MIXTURE LOG DENSITY DISTRIBUTION (KDE')
            plt.show()

            print(pd.DataFrame({'predict proba index': prob_index,
                                'predict proba count': prob_value}).describe())


        except Exception as e:
            raise RuntimeError(f'invalid gaussian mixture: {e}')


if __name__ == "__main__":
    abc = ABC()
    abc.b()
