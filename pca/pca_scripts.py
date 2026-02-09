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

            pipe = Pipeline([('preprocessor', self.preprocessor),
                             ('pca', PCA(n_components=0.9, svd_solver='auto',random_state=42))])

            x = pipe.named_steps['preprocessor'].fit_transform(self.copy)
            x_pca = pipe.named_steps['pca'].fit_transform(x)

            variance = pipe.named_steps['pca'].explained_variance_
            variance_ratio = pipe.named_steps['pca'].explained_variance_ratio_

            c_sum = np.cumsum(variance)
            c_sum_ratio = np.cumsum(variance_ratio)

            v_index = pd.Series(variance).index.to_numpy()
            v_value = pd.Series(variance).to_numpy()

            vr_index = pd.Series(variance_ratio).index.to_numpy()
            vr_value = pd.Series(variance_ratio).to_numpy()

            c_index = pd.Series(c_sum).index.to_numpy()
            c_value = pd.Series(c_sum).to_numpy()

            cr_index = pd.Series(c_sum_ratio).index.to_numpy()
            cr_value = pd.Series(c_sum_ratio).to_numpy()

            plt.figure(figsize=(10,8))
            plt.scatter(v_index, v_value, c='red', s=40,label='Variance')
            plt.scatter(v_index, vr_value, c='green', s=40,label='Variance ratio')
            plt.legend()
            plt.title('VARIANCE / VARIANCE RATIO')
            plt.xlabel('INDEX')
            plt.ylabel('VALUE')
            plt.show()

            plt.figure(figsize=(10,8))
            plt.scatter(c_index, c_value,c='red', s=40, label='Cumsum (Variance)')
            plt.scatter(c_index, cr_value, c='green', s=40, label='Cumsum (Variance ratio)')
            plt.legend()
            plt.title('CUMSUM (VARIANCE) / CUMSUM (VARIANCE RATIO)')
            plt.xlabel('INDEX')
            plt.ylabel('VALUE')
            plt.show()

            inverse_pipe = Pipeline([('preprocessor', self.preprocessor),
                                     ('pca', PCA(n_components=0.9, svd_solver='auto', random_state=42))])

            i_x = inverse_pipe.named_steps['preprocessor'].fit_transform(self.copy)
            i_pca = inverse_pipe.named_steps['pca'].fit_transform(i_x)

            inverse_xpca = inverse_pipe.named_steps['pca'].inverse_transform(i_pca)

            plt.figure(figsize=(10,8))
            plt.scatter(i_x[:,0], i_x[:,1], c='red', s=40, label='Original')
            plt.scatter(inverse_xpca[:,0], inverse_xpca[:,1], c='green', s=30, label='Inverse PCA')
            plt.legend()
            plt.title('ORIGINAL FEATURES 0 - 1 INVERSE FEATURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.show()

            plt.figure(figsize=(10, 8))
            plt.scatter(i_x[:,2], i_x[:,3], c='red', s=40, label='Original')
            plt.scatter(inverse_xpca[:,2], inverse_xpca[:,3], c='green', s=30, label='Inverse PCA')
            plt.legend()
            plt.title('ORIGINAL FEATURES 2 - 3 INVERSE FEATURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.show()

            plt.figure(figsize=(10, 8))
            plt.scatter(i_x[:, 4], i_x[:, 5], c='red', s=40, label='Original')
            plt.scatter(inverse_xpca[:, 4], inverse_xpca[:, 5], c='green', s=30, label='Inverse PCA')
            plt.legend()
            plt.title('ORIGINAL FEATURES 4 - 5 INVERSE FEATURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.show()

            plt.figure(figsize=(10, 8))
            plt.scatter(i_x[:, 6], i_x[:, 7], c='red', s=40, label='Original')
            plt.scatter(inverse_xpca[:, 6], inverse_xpca[:, 7], c='green', s=30, label='Inverse PCA')
            plt.legend()
            plt.title('ORIGINAL FEATURES 6 - 7 INVERSE FEATURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.show()

            plt.figure(figsize=(10, 8))
            plt.scatter(i_x[:, 8], i_x[:, 9], c='red', s=40, label='Original')
            plt.scatter(inverse_xpca[:, 8], inverse_xpca[:, 9], c='green', s=30, label='Inverse PCA')
            plt.legend()
            plt.title('ORIGINAL FEATURES 8 - 9 INVERSE FEATURES')
            plt.xlabel('INDEX')
            plt.ylabel('VALUES')
            plt.show()

            v_c_sum = pd.DataFrame({'variance': pd.Series(MinMaxScaler().fit_transform(variance.reshape(-1,1)).ravel()),
                                    'variance ratio': pd.Series(MinMaxScaler().fit_transform(variance_ratio.reshape(-1,1)).ravel()),
                                    'cumsum': pd.Series(MinMaxScaler().fit_transform(c_sum.reshape(-1,1)).ravel()),
                                    'cumsum ratio': pd.Series(MinMaxScaler().fit_transform(c_sum_ratio.reshape(-1,1)).ravel())})

            print(v_c_sum.describe())

        except Exception as e:
            raise RuntimeError(f'invalid PCA: {e}')




if __name__ == "__main__":
    ab = AB()
    ab.b()
