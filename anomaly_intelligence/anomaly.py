import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import hdbscan
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class A:
    def __init__(self):
        self.df = pd.read_csv('c:/Users/anton/risk/credit.csv', encoding='utf-8-sig', engine='python')
        self.df.columns = self.df.columns.str.replace('_', ' ', regex=True).str.lower().str.strip()

        self.df = self.df.drop(columns=['loan grade', 'loan status', 'cb person default on file'])

        self.df = self.df.rename(columns={'person age': 'age',
                                'person income': 'income',
                                'person home ownership': 'owner',
                                'person emp length': 'length',
                                'loan intent': 'intent',
                                'loan amnt': 'amount',
                                'loan int rate': 'interest',
                                'loan percent income': 'percent',
                                'cb person cred hist length': 'card'})

        self.df.columns = self.df.columns.str.lower().str.strip()

        self.copy = self.df.copy()

        self.min = MinMaxScaler()
        self.one = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        self.num = self.copy.select_dtypes(include=['number']).columns
        self.obj = self.copy.select_dtypes(include=['object']).columns

        n_simple = SimpleImputer(strategy='median')
        o_simple = SimpleImputer(strategy='most_frequent')

        num_pipeline = Pipeline([('num', n_simple),
                                 ('scaler', self.min)])
        obj_pipeline = Pipeline([('obj', o_simple),
                                 ('encoder', self.one)])

        self.preprocessor = ColumnTransformer([('n', num_pipeline, self.num),
                                               ('o', obj_pipeline, self.obj)])

        self.copy = self.preprocessor.fit_transform(self.copy)

        self.scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

    def b(self):

        dbscan = DBSCAN(eps=0.2, min_samples=7)
        pca = PCA(n_components=0.9)

        x_pca = pca.fit_transform(self.copy)
        dbscan.fit(x_pca)
        db_label = dbscan.labels_

        for score in self.scores:
            s = score(x_pca, db_label)
            print(f'{score.__name__}: {s}')
        print()

        self.df['dbscan label'] = db_label

        db_size = pd.Series(db_label).value_counts()
        self.df['dbscan cluster size'] = pd.Series(db_label).map(db_size).values

        db_invert = self.df['dbscan cluster size'].max() - self.df['dbscan cluster size']

        self.df['dbscan cluster score'] = self.min.fit_transform(db_invert.values.reshape(-1,1)).ravel()

        top = self.df['dbscan cluster score'].quantile(0.75)
        mi = self.df['dbscan cluster score'].quantile(0.25)

        self.df['dbscan anomaly'] = np.where((self.df['dbscan cluster score'] >= top), 'High',
                                             np.where((self.df['dbscan cluster score'] >= mi), 'Medium', 'Low'))

        op = OPTICS(min_samples=3, xi=0.9)

        op.fit(x_pca)
        op_tics = op.labels_

        for o in self.scores:
            o_score = o(x_pca, op_tics)
            print(f'{o.__name__}: {o_score}')
        print()

        reach = op.reachability_.copy()

        reach_convert = reach[np.isfinite(reach)].max()
        reach = np.where(np.isinf(reach), reach_convert, reach)
        self.df['optics reachability score'] = reach

        hi = self.df['optics reachability score'].quantile(0.75)
        med = self.df['optics reachability score'].quantile(0.25)

        self.df['optics anomaly'] = np.where((self.df['optics reachability score'] >= hi), 'High',
                                             np.where((self.df['optics reachability score'] >= med), 'Medium', 'Low'))

        hd = hdbscan.HDBSCAN(min_samples=8, min_cluster_size= 7)
        hd.fit(x_pca)
        hd_label = hd.labels_

        for hd_score in self.scores:
            h = hd_score(x_pca, hd_label)
            print(f'{hd_score.__name__}: {h}')
        print()

        self.df['hdbscan label'] = hd_label

        prob = hd.probabilities_

        self.df['hdbscan probability'] = prob

        h = self.df['hdbscan probability'].quantile(0.75)
        m = self.df['hdbscan probability'].quantile(0.25)

        self.df['hdbscan confidence'] = np.where((self.df['hdbscan probability'] >= h), 'High',
                                                 np.where((self.df['hdbscan probability'] >= m), 'Medium', 'Low'))

        self.df['hdbscan outlier score'] = hd.outlier_scores_

        top = self.df['hdbscan outlier score'].quantile(0.75)
        middle = self.df['hdbscan outlier score'].quantile(0.25)

        self.df['hdbscan anomaly'] = np.where((self.df['hdbscan outlier score'] >= top), 'High',
                                              np.where((self.df['hdbscan outlier score'] >= middle), 'Medium', 'Low'))

        lof = LocalOutlierFactor(n_neighbors=100)

        lof.fit(x_pca)
        lof_predict = lof.fit_predict(x_pca)

        neg_outlier = lof.negative_outlier_factor_

        self.df['local label'] = lof_predict

        invert_local = neg_outlier.reshape(-1,1)

        local_scale = self.min.fit_transform(invert_local).ravel()

        self.df['local score'] = 1 - local_scale

        local_max = self.df['local score'].quantile(0.75)
        local_med = self.df['local score'].quantile(0.25)

        self.df['local anomaly'] = np.where((self.df['local score'] >= local_max), 'High',
                                            np.where((self.df['local score'] >= local_med), 'Medium', 'Low'))


        self.df['combine anomaly'] = np.where(
            # Top condition → all High
            (self.df['dbscan anomaly'] == 'High') &
            (self.df['optics anomaly'] == 'High') &
            (self.df['hdbscan confidence'] == 'High') &
            (self.df['hdbscan anomaly'] == 'High') &
            (self.df['local anomaly'] == 'High'),
            'High',

            np.where(
                (self.df['dbscan anomaly'].isin(['Medium', 'Low'])) &
                (self.df['optics anomaly'].isin(['Medium', 'Low'])) &
                (self.df['hdbscan confidence'].isin(['Medium', 'Low'])) &
                (self.df['hdbscan anomaly'].isin(['Medium', 'Low'])) &
                (self.df['local anomaly'].isin(['Medium', 'Low'])),
                'Medium',
                'Low'))

        iso = IsolationForest(n_estimators=100)

        iso.fit(x_pca)
        pre_iso = iso.predict(x_pca)
        iso_decision = iso.decision_function(x_pca)

        self.df['isolation label'] = pre_iso

        invert_iso = iso_decision.reshape(-1,1)
        iso_min = self.min.fit_transform(invert_iso).ravel()

        self.df['isolation score'] = 1 - iso_min

        iso_max = self.df['isolation score'].quantile(0.75)
        iso_med = self.df['isolation score'].quantile(0.25)

        self.df['isolation anomaly'] = np.where((self.df['isolation score'] >= iso_max), 'High', np.where((self.df['isolation score'] >= iso_med), 'Medium', 'Low'))

        self.df['final iso combine alert'] = np.where((self.df['combine anomaly'] == 'High') & (self.df['isolation anomaly'] == 'High'), 'High',
                                                      np.where((self.df['combine anomaly'].isin(['Medium', 'Low'])) & (self.df['isolation anomaly'].isin(['Medium', 'Low'])), 'Medium', 'Low'))

        svm = OneClassSVM()

        svm.fit(x_pca)
        svm_predict = svm.predict(x_pca)
        svm_decision = svm.decision_function(x_pca)

        svm_score = svm_decision.reshape(-1, 1)
        svm_ravel = self.min.fit_transform(svm_score).ravel()

        svm_final = 1 - svm_ravel

        self.df['svm score'] = svm_final

        svm_max = self.df['svm score'].quantile(0.75)
        svm_med = self.df['svm score'].quantile(0.25)

        self.df['svm anomaly'] = np.where((self.df['svm score'] >= svm_max), 'High',
                                          np.where((self.df['svm score'] >= svm_med), 'Medium', 'Low'))

        self.df['final svm combine alert'] = np.where((self.df['combine anomaly'] == 'High') & (self.df['svm anomaly'] == 'High'), 'High',
                                                      np.where((self.df['combine anomaly'].isin(['Medium', 'Low'])) & (self.df['svm anomaly'].isin(['Medium', 'Low'])), 'Medium', 'Low'))

        self.df['final notification'] = np.where((self.df['final iso combine alert'] == 'High') & (self.df['final svm combine alert'] == 'High'), 'High',
                                                 np.where((self.df['final iso combine alert'].isin(['Medium', 'Low'])) & (self.df['final svm combine alert'].isin(['Medium', 'Low'])), 'Medium', 'Low'))

        self.df = self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index+1)
        print(self.df.head(5).to_string())

        #self.df.to_csv('c:/Users/anton/risk/credit_final.csv', index=False)


if __name__ == "__main__":
    a = A()
    a.b()
