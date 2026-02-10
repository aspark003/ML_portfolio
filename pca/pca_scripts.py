import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class AB:
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

            x= self.preprocessor.fit_transform(self.copy)
            pca = PCA(n_components=0.9)

            x_pca = pca.fit_transform(x)

            variance_ratio = pca.explained_variance_ratio_
            cumsum = np.cumsum(variance_ratio)

            v = np.arange(len(variance_ratio))

            plt.figure(figsize=(10,8))
            plt.scatter(v, variance_ratio, c='red',s=40, label='VARIANCE RATIO')
            plt.scatter(v, cumsum, c='green', s=30, label='CUMULATIVE')
            plt.title('VARIANCE RATIO VS CUMULATIVE')
            plt.xlabel('VARIANCE INDEX')
            plt.ylabel('VALUES')
            plt.legend()
            plt.show()

            component = pca.components_
            len_com = np.arange(len(component))

            plt.figure(figsize=(10,8))
            plt.scatter(component[:,0], component[:,1], c='red', s=40)
            plt.scatter(component[:,1], component[:,2], c='green',s=30)
            plt.xlabel('FEATURE 0')
            plt.ylabel('FEATURES 1')
            plt.title('PCA PAIR COMPARISONS')
            plt.show()

            inverse_pca = pca.inverse_transform(x_pca)

            plt.figure(figsize=(10,8))
            plt.scatter(inverse_pca[:,0], inverse_pca[:,3], c='red', s=50, label='INVERSE PCA')
            plt.scatter(x[:,0], x[:,3], c='green', s=30, label='ORIGINAL X')
            plt.xlabel('FEATURE 0 - 3')
            plt.ylabel('FEATURES 0 - 3')
            plt.title('INVERSE PCA VS ORIGINAL X')
            plt.legend()
            plt.show()

            vc_signal = pd.DataFrame({'variance ratio': variance_ratio,
                                      'cumulative': cumsum})
            print(vc_signal.describe())

            print(pd.DataFrame(x_pca, columns= [f'PC:{i+1}' for i in range(x_pca.shape[1])]).describe().to_string())

            print(pd.DataFrame(component, columns=[f' COMPONENT:{i+1}' for i in range(component.shape[1])]).describe().to_string())

        except Exception as e:
            raise RuntimeError(f'invalid PCA: {e}')




if __name__ == "__main__":
    ab = AB()
    ab.b()
