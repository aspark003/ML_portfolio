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

        hd = Pipeline([('preprocessor', preprocessor),
                       ('pca', PCA(n_components=0.8)),
                       ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=7, copy=False))])

        iso = Pipeline([('preprocessor', preprocessor),
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
            print(f'dbscan: {sc.__name__}:{s}')
        print()

        hd.fit(self.df)
        xh = hd.named_steps['preprocessor'].transform(self.df)
        xh_pca = hd.named_steps['pca'].transform(xh)
        yh = hd.named_steps['hdbscan'].labels_
        for ys in scores:
            ysco = ys(xh_pca, yh)
            print(f'hdbscan: {ys.__name__}:{ysco}')
        print()

        self.df['hdbscan label'] = yh

        self.df['outlier flag'] = self.df['hdbscan label'] == -1

        prob = hd.named_steps['hdbscan'].probabilities_

        self.df['probability score'] = - prob
        self.df['probability score scaled'] = MinMaxScaler().fit_transform(self.df['probability score'].values.reshape(-1,1)).ravel()

        self.df['probability level'] = np.where((self.df['outlier flag'] == True) & (self.df['probability score scaled'] > 0.75),'high', np.where((self.df['outlier flag'] == True) & (self.df['probability score scaled'] >= 0.25), 'medium','low'))

        iso.fit(self.df)
        x_decision = iso.named_steps['preprocessor'].transform(self.df)
        x_dec_predict = iso.named_steps['iso'].predict(x_decision)
        x_dec_function = iso.named_steps['iso'].decision_function(x_decision)

        self.df['isolation prediction'] = x_dec_predict
        self.df['isolation outlier flag'] = (self.df['isolation prediction'] == -1)
        self.df['isolation risk score'] = - x_dec_function

        self.df['isolation risk score scaled'] = MinMaxScaler().fit_transform(self.df['isolation risk score'].values.reshape(-1,1)).ravel()

        self.df['isolation level'] = np.where((self.df['isolation outlier flag'] == True) & (self.df['isolation risk score scaled'] > 0.75), 'high', np.where((self.df['isolation outlier flag'] == True) & (self.df['isolation risk score scaled'] >= 0.25), 'medium','low'))

        self.df["Severity Level"] = np.select([(self.df["probability level"] == "high") & (self.df["isolation level"] == "high"),
                (self.df["probability level"].isin(["high", "medium"])) & (self.df["isolation level"].isin(["high", "medium"]))],["high", "medium"],default="low")

        self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)
        print(self.df.head(30).to_string())
        print()
        #self.df.to_csv('c:/Users/anton/unsuper/cluster/final.csv', index=False)
        #print(self.df.head(50).to_string())

        #print(iso_d)


if __name__ == "__main__":
    a = A('c:/Users/anton/unsuper/cluster/copy_raw.csv')
    a.b()

