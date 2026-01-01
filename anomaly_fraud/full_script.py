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
                       ('dbscan', DBSCAN(eps=0.2, min_samples=5))])

        op = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.9)),
                       ('optics', OPTICS(min_samples=5, xi=0.05))])

        hd = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.9)),
                       ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=5, copy=False))])

        iso = Pipeline([('preprocessor', preprocessor),
                        ('pca', PCA(n_components=0.9)),
                        ('iso', IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', random_state=42))])
        scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

        db.fit(self.df)
        x = db.named_steps['preprocessor'].transform(self.df)
        x_pca = db.named_steps['pca'].transform(x)
        variance = db.named_steps['pca'].explained_variance_ratio_
        cumsum = np.cumsum(variance)
        print(variance)
        print(cumsum)
        y = db.named_steps['dbscan'].labels_
        for sc in scores:
            s = sc(x_pca, y)
            print(f'{sc.__name__}:{s}')
        hd.fit(self.df)
        xh = hd.named_steps['preprocessor'].transform(self.df)
        xh_pca = hd.named_steps['pca'].transform(xh)
        yh = hd.named_steps['hdbscan'].labels_
        for ys in scores:
            ysco = ys(xh_pca, yh)
            print(f'{ys.__name__}:{ysco}')

        prob = hd.named_steps['hdbscan'].probabilities_


        self.df['dbscan label'] = y
        self.df['outlier'] = (self.df['dbscan label'] == -1).astype(int)
        self.df['outlier label'] = (self.df['outlier'].apply(lambda x: 'review' if x == 1 else 'no issues'))

        self.df['risk score'] = prob

        med = 0.5

        self.df['risk level'] = (self.df['risk score'].apply(lambda x: 'high' if x <= med else 'low'))

        iso.fit(self.df)
        ix = iso.named_steps['preprocessor'].transform(self.df)
        ix_pca = iso.named_steps['pca'].transform(ix)
        iso_d = iso.named_steps['iso'].decision_function(ix_pca)

        self.df['anomaly score'] = iso_d
        middle = 0.5
        self.df['anomaly detector'] = (self.df['anomaly score'].apply(lambda x: 'critical' if x <= middle else 'low'))
        self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)
        #self.df.to_csv('c:/Users/anton/unsuper/cluster/final.csv', index=False)
        print(self.df.head().to_string())

        #print(iso_d)


if __name__ == "__main__":
    a = A('c:/Users/anton/unsuper/cluster/copy_raw.csv')
    a.b()

