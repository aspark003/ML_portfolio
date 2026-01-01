import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class A:
    def __init__(self, file):
        self.df = pd.read_csv(file, encoding='utf-8-sig', engine='python')

    def b(self):
        self.df.columns =self.df.columns.str.replace('_', ' ', regex=True).str.lower().str.strip()

        num = self.df.select_dtypes(include=['number']).columns
        obj = self.df.select_dtypes(include=['object', 'string']).columns

        scaler = MinMaxScaler()
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        n = SimpleImputer(strategy='median')
        o = SimpleImputer(strategy='most_frequent')

        ns = Pipeline([('n', n),
                       ('scaler', scaler)])
        os = Pipeline([('o',o),
                       ('encoder', encoder)])


        preprocessor = ColumnTransformer([('num', ns, num),
                                          ('obj', os, obj)])

        db = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.9)),
                       ('dbscan', DBSCAN(eps=0.4, min_samples=8))])

        op = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.8)),
                       ('optics', OPTICS(min_samples=6, xi=0.01))])

        hd = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.8)),
                       ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=7, copy=False))])

        iso = Pipeline([('preprocessor', preprocessor),
                        ('pca', PCA(n_components=0.9)),
                        ('iso', IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42))])
        scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

        db.fit(self.df)
        x = db.named_steps['preprocessor'].transform(self.df)
        x_pca = db.named_steps['pca'].transform(x)
        variance = db.named_steps['pca'].explained_variance_ratio_
        cumsum = np.cumsum(variance)
        #print(variance)
        #print(cumsum)
        y = db.named_steps['dbscan'].labels_
        for sc in scores:
            s = sc(x_pca, y)
            print(f'dbscan: {sc.__name__}:{s}')
        print()


        op.fit(self.df)
        ox = op.named_steps['preprocessor'].transform(self.df)
        ox_pca = op.named_steps['pca'].transform(ox)
        oy = op.named_steps['optics'].labels_
        for oyscore in scores:
            oys = oyscore(ox_pca, oy)
            print(f'optics: {oyscore.__name__}:{oys}')
        print()

        hd.fit(self.df)
        xh = hd.named_steps['preprocessor'].transform(self.df)
        xh_pca = hd.named_steps['pca'].transform(xh)
        yh = hd.named_steps['hdbscan'].labels_
        for ys in scores:
            ysco = ys(xh_pca, yh)
            print(f'hdbscan: {ys.__name__}:{ysco}')
        print()

        prob = hd.named_steps['hdbscan'].probabilities_


        self.df['hdbscan label'] = yh
        self.df['outlier'] = (self.df['hdbscan label'].apply(lambda x: 1 if x == -1 else 0))
        self.df['outlier label'] = (self.df['outlier'].apply(lambda x: 'review' if x == 1 else 'no issues'))

        self.df['risk score'] = prob

        threshold = self.df['risk score'].quantile(0.25)

        self.df['risk level'] = np.where(self.df['risk score'] <= threshold, 'high', 'low')
        iso.fit(self.df)
        ix = iso.named_steps['preprocessor'].transform(self.df)
        ix_pca = iso.named_steps['pca'].transform(ix)
        iso_d = iso.named_steps['iso'].decision_function(ix_pca)

        self.df['anomaly score'] = iso_d

        cutoff = self.df['anomaly score'].quantile(0.25)


        self.df['anomaly detector'] = np.where(self.df['anomaly score'] <= cutoff, 'critical', 'low')
        self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)
        #self.df.to_csv('c:/Users/anton/unsuper/cluster/final.csv', index=False)
        print(self.df.head(1).to_string())

        #print(iso_d)


if __name__ == "__main__":
    a = A('c:/Users/anton/unsuper/cluster/copy_raw.csv')
    a.b()

