import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class A:
    def __init__(self, file):
        self.df = pd.read_csv(file, encoding='utf-8-sig', engine='python')

        self.copy = self.df.copy()

        self.num = self.copy.select_dtypes(include=['number']).columns
        self.obj = self.copy.select_dtypes(include=['object', 'string']).columns

        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        self.s_simple = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                 ('scaler', self.scaler)])

        self.e_simple = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encoder', self.encoder)])

        self.preprocessor = ColumnTransformer([('scaler', self.s_simple, self.num),
                                               ('encoder', self.e_simple, self.obj)])

    def b(self):

        x = self.preprocessor.fit_transform(self.copy)
        pca = PCA(n_components=0.9)

        x_pca = pca.fit_transform(x)

        iso = IsolationForest(random_state=42)

        iso.fit(x_pca)
        x_iso = iso.predict(x_pca)

        d_iso = iso.decision_function(x_pca)

        self.df['Isolation labels'] = x_iso

        self.df['Isolation scores'] = d_iso

        mm = MinMaxScaler()
        mm1 = MinMaxScaler()
        mm2 = MinMaxScaler()

        mm_scale = d_iso.reshape(-1,1)

        mm_up = mm.fit_transform(mm_scale).ravel()

        self.df['Isolation risk scores'] = 1 - mm_up

        self.df['Isolation risk level'] = np.where((self.df['Isolation risk scores'] >= 0.75), 'Critical', np.where((self.df['Isolation risk scores'] >= 0.25), 'High', 'Low'))

        local = LocalOutlierFactor(n_neighbors=100)

        x_l = local.fit_predict(x_pca)

        factor = local.negative_outlier_factor_

        self.df['Local Outlier Labels'] = x_l
        self.df['Local Outlier Scores'] = factor

        self.df['Local Risk Level'] = np.where(
            (self.df['Local Outlier Scores'] <= self.df['Local Outlier Scores'].quantile(0.25)), 'Critical',
            np.where((self.df['Local Outlier Scores'] <= self.df['Local Outlier Scores'].quantile(0.75)), 'High',
                     'Low'))


        pca_recon = pca.inverse_transform(x_pca)
        recon_error = np.mean((x - pca_recon) **2, axis=1)

        self.df['PCA Error'] = recon_error

        p_sha = recon_error.reshape(-1, 1)

        p_scaled = mm1.fit_transform(p_sha).ravel()

        self.df['Scaled PCA'] = p_scaled

        self.df['PCA Level'] = np.where((self.df['Scaled PCA'] >= 0.75), 'Critical',
                                        np.where((self.df['Scaled PCA'] >= 0.25), 'High', 'Low'))

        svm = OneClassSVM()
        svm.fit(x_pca)
        svm_predict = svm.predict(x_pca)

        svm_decision = svm.decision_function(x_pca)

        self.df['SVM labels'] = svm_predict


        svm_s = svm_decision.reshape(-1,1)
        svm_scaled = mm2.fit_transform(svm_s).ravel()

        svm_risk = 1 - svm_scaled
        self.df['SVM risk scores'] = svm_risk
        self.df['SVM risk level'] = np.where((self.df['SVM risk scores'] >= 0.75), 'Critical', np.where((self.df['SVM risk scores'] >= 0.25), 'High', 'Low'))


        self.df['Severity Level'] = np.where(
            (self.df['Isolation risk level'] == 'Critical') &
            (self.df['Local Risk Level'] == 'Critical') &
            (self.df['PCA Level'] == 'Critical') &
            (self.df['SVM risk level'] == 'Critical'),
            'Critical',
            np.where(
                (self.df['Isolation risk level'].isin(['Critical', 'High'])) &
                (self.df['Local Risk Level'].isin(['Critical', 'High'])) &
                (self.df['PCA Level'].isin(['Critical', 'High'])) &
                (self.df['SVM risk level'].isin(['Critical', 'High'])),
                'High',
                'Low'))

        self.df = self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)

        #self.df.to_csv('c:/Users/anton/unsuper/cluster/final_powerbi.csv', index=False)
        print(self.df.head().to_string())

if __name__ == "__main__":
    a = A('c:/Users/anton/unsuper/cluster/copy_raw.csv')
    a.b()


