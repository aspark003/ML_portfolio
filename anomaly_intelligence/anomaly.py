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

        self.df['dbscan severity level'] = np.where((self.df['dbscan cluster score'] >= top), 'High',
                                             np.where((self.df['dbscan cluster score'] > mi), 'Medium', 'Low'))

        op = OPTICS(min_samples=3, xi=0.9)

        op.fit(x_pca)
        op_tics = op.labels_

        self.df['optics label'] = op_tics

        for o in self.scores:
            o_score = o(x_pca, op_tics)
            print(f'{o.__name__}: {o_score}')
        print()

        reach = op.reachability_.copy()

        self.df['reachability cluster size'] = reach

        reach_convert = reach[np.isfinite(reach)].max()
        reach = np.where(np.isinf(reach), reach_convert, reach)

        invert_reach = reach.reshape(-1, 1)

        reach_scale = self.min.fit_transform(invert_reach).ravel()

        self.df['optics reachability cluster size'] = reach

        self.df['optics reachability scaled score'] = reach_scale

        hi = self.df['optics reachability scaled score'].quantile(0.75)
        med = self.df['optics reachability scaled score'].quantile(0.25)

        self.df['optics severity level'] = np.where((self.df['optics reachability scaled score'] >= hi), 'High',
                                             np.where((self.df['optics reachability scaled score'] > med), 'Medium', 'Low'))

        hd = hdbscan.HDBSCAN(min_samples=8, min_cluster_size= 7)
        hd.fit(x_pca)
        hd_label = hd.labels_

        for hd_score in self.scores:
            h = hd_score(x_pca, hd_label)
            print(f'{hd_score.__name__}: {h}')
        print()

        self.df['hdbscan label'] = hd_label

        prob = hd.probabilities_.copy()

        self.df['hdbscan probability'] = prob

        h = self.df['hdbscan probability'].quantile(0.75)
        m = self.df['hdbscan probability'].quantile(0.25)

        self.df['hdbscan probability confidence'] = np.where((self.df['hdbscan probability'] >= h), 'High',
                                                 np.where((self.df['hdbscan probability'] > m), 'Medium', 'Low'))

        self.df['hdbscan probability severity score'] = 1 - self.df['hdbscan probability']

        hp_max = self.df['hdbscan probability severity score'].quantile(0.75)
        hp_med = self.df['hdbscan probability severity score'].quantile(0.25)

        self.df['hdbscan probability severity level'] = np.where(
            self.df['hdbscan probability severity score'] >= hp_max, 'High',
            np.where(self.df['hdbscan probability severity score'] > hp_med, 'Medium', 'Low'))

        self.df['hdbscan outlier score'] = hd.outlier_scores_.copy()

        top = self.df['hdbscan outlier score'].quantile(0.75)
        middle = self.df['hdbscan outlier score'].quantile(0.25)

        self.df['hdbscan outlier severity level'] = np.where((self.df['hdbscan outlier score'] >= top), 'High',
                                              np.where((self.df['hdbscan outlier score'] > middle), 'Medium', 'Low'))

        lof = LocalOutlierFactor(n_neighbors=100)

        lof_predict = lof.fit_predict(x_pca)

        self.df['local outlier factor label'] = lof_predict

        neg_outlier = lof.negative_outlier_factor_.copy()

        self.df['local outlier factor score'] = neg_outlier

        invert_local = neg_outlier.reshape(-1,1)

        local_scale = self.min.fit_transform(invert_local).ravel()

        self.df['local outlier severity score'] = 1 - local_scale

        local_max = self.df['local outlier severity score'].quantile(0.75)
        local_med = self.df['local outlier severity score'].quantile(0.25)

        self.df['local outlier severity level'] = np.where((self.df['local outlier severity score'] >= local_max), 'High',
                                            np.where((self.df['local outlier severity score'] > local_med), 'Medium', 'Low'))

        self.df['density severity level'] = np.where(
            (self.df['dbscan severity level'] == 'High') &
            (self.df['optics severity level'] == 'High') &
            (self.df['hdbscan probability severity level'] == 'High') &
            (self.df['hdbscan outlier severity level'] == 'High') &
            (self.df['local outlier severity level'] == 'High'),
            'High',
            np.where(
                (self.df['dbscan severity level'].isin(['High', 'Medium'])) &
                (self.df['optics severity level'].isin(['High', 'Medium'])) &
                (self.df['hdbscan probability severity level'].isin(['High', 'Medium'])) &
                (self.df['hdbscan outlier severity level'].isin(['High', 'Medium'])) &
                (self.df['local outlier severity level'].isin(['High', 'Medium'])),
                'Medium',
                'Low'))

        iso = IsolationForest(n_estimators=100)

        pre_iso = iso.fit_predict(x_pca)

        self.df['isolation label'] = pre_iso

        iso_decision = iso.decision_function(x_pca)

        self.df['isolation decision score'] = iso_decision

        invert_iso = iso_decision.reshape(-1, 1)
        iso_min = self.min.fit_transform(invert_iso).ravel()

        self.df['isolation severity score'] = 1 - iso_min

        iso_max = self.df['isolation severity score'].quantile(0.75)
        iso_med = self.df['isolation severity score'].quantile(0.25)

        self.df['isolation severity level'] = np.where((self.df['isolation severity score'] >= iso_max), 'High',
                                                np.where((self.df['isolation severity score'] > iso_med), 'Medium', 'Low'))

        self.df['final density severity level'] = np.where((self.df['density severity level'] == 'High') & (self.df['isolation severity level'] == 'High'), 'High',
                                                           np.where((self.df['density severity level'].isin(['High', 'Medium'])) & (self.df['isolation severity level'].isin(['High', 'Medium'])), 'Medium', 'Low'))

        svm = OneClassSVM()

        svm_predict = svm.fit_predict(x_pca)

        self.df['svm label'] = svm_predict

        svm_decision = svm.decision_function(x_pca)

        self.df['svm decision score'] = svm_decision

        svm_score = svm_decision.reshape(-1, 1)
        svm_ravel = self.min.fit_transform(svm_score).ravel()

        svm_final = 1 - svm_ravel

        self.df['svm severity score'] = svm_final

        svm_max = self.df['svm severity score'].quantile(0.75)
        svm_med = self.df['svm severity score'].quantile(0.25)

        self.df['svm severity level'] = np.where((self.df['svm severity score'] >= svm_max), 'High',
                                          np.where((self.df['svm severity score'] > svm_med), 'Medium', 'Low'))

        self.df['severity level'] = np.where((self.df['svm severity level'] == 'High') & (self.df['final density severity level'] == 'High'), 'High',
                                             np.where((self.df['svm severity level'].isin(['High', 'Medium'])) & (self.df['final density severity level'].isin(['High', 'Medium'])), 'Medium', 'Low'))

        self.df = self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index+1)
        print(self.df.head(5).to_string())

        #self.df.to_csv('c:/Users/anton/risk/credit_final.csv', index=False)


if __name__ == "__main__":
    a = A()
    a.b()
