import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor

class A:
    def __init__(self):
        self.df = pd.read_csv('c:/Users/anton/projects/fraud.csv')
        self.copy = self.df.copy()
        self.copy.columns = self.df.columns.str.lower().str.strip()
        self.copy = self.copy.drop(columns=['time', 'class'])

    def b(self):
        pca = PCA(n_components=0.7)

        dbscan = IsolationForest()
        x = pca.fit_transform(self.copy)

        dbscan.fit(x)
        y = dbscan.predict(x)

        decision = dbscan.decision_function(x)

        self.df['Isolation labels'] = y
        self.df['Decision Scores'] = decision

        mm = MinMaxScaler()
        mm1 = MinMaxScaler()

        c = decision.reshape(-1,1)

        scaled = mm.fit_transform(c).ravel()

        self.df['Risk Score'] = 1 - scaled

        self.df['Risk Level'] = np.where((self.df['Risk Score'] >= 0.75), 'Critical', np.where((self.df['Risk Score'] >= 0.25), 'High', 'Low'))

        local = LocalOutlierFactor(n_neighbors=100)

        x_l = local.fit_predict(x)

        factor = local.negative_outlier_factor_

        self.df['Local Outlier Labels'] = x_l
        self.df['Local Outlier Scores'] = factor

        self.df['Local Risk Level'] = np.where((self.df['Local Outlier Scores'] <= self.df['Local Outlier Scores'].quantile(0.25)),'Critical',
                                               np.where((self.df['Local Outlier Scores'] <= self.df['Local Outlier Scores'].quantile(0.75)), 'High', 'Low'))

        pca_recon = pca.inverse_transform(x)
        recon_error = np.mean((self.copy.values - pca_recon) ** 2, axis=1)

        self.df['PCA Error'] = recon_error

        p_sha = recon_error.reshape(-1,1)

        p_scaled = mm1.fit_transform(p_sha).ravel()

        self.df['Scaled PCA'] = p_scaled

        self.df['PCA Level'] = np.where((self.df['Scaled PCA'] >= 0.75), 'Critical', np.where((self.df['Scaled PCA'] >= 0.25), 'High', 'Low'))

        self.df['Severity Level'] = np.where(
            (self.df['Risk Level'] == 'Critical') &
            (self.df['Local Risk Level'] == 'Critical') &
            (self.df['PCA Level'] == 'Critical'),
            'Critical',
            np.where(
                (self.df['Risk Level'].isin(['Critical', 'High'])) &
                (self.df['Local Risk Level'].isin(['Critical', 'High'])) &
                (self.df['PCA Level'].isin(['Critical', 'High'])),
                'High',
                'Low'))

        self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index +1)

        #self.df.to_csv('c:/Users/anton/projects/full_detector.csv', index=False)
        print(self.df.head(10).to_string())

if __name__ == "__main__":
    A = A()
    A.b()
