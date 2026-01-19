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
        self.df = pd.read_csv('c:/Users/anton/risk/draft.csv', encoding='utf-8-sig', engine='python')

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

        self.df['dbscan label'] = db_label

        self.df['dbscan label size'] = self.df['dbscan label'].map(self.df['dbscan label'].value_counts())

        mm1 = MinMaxScaler()
        mm0 = MinMaxScaler()

        self.df['dbscan confidence score'] = mm1.fit_transform(self.df['dbscan label size'].to_numpy().reshape(-1,1)).ravel()

        db_scale = 1 - self.df['dbscan confidence score']

        self.df['dbscan severity score'] = mm0.fit_transform(db_scale.to_numpy().reshape(-1,1)).ravel()

        db_top = self.df['dbscan severity score'].quantile(0.75)
        db_mid = self.df['dbscan severity score'].quantile(0.25)

        self.df['dbscan severity level'] = np.select([self.df['dbscan severity score'] >= db_top,
                                                      (self.df['dbscan severity score'] > db_mid) & (self.df['dbscan severity score'] < db_top)],['High', 'Medium'], default='Low')

        op = OPTICS(min_samples = 3, xi=0.9)

        op.fit(x_pca)
        op_tics = op.labels_

        self.df['optics label'] = op_tics

        reach = op.reachability_.copy()

        self.df['reachability'] = reach

        max_reach = reach[np.isfinite(reach)].max()

        inf_reach = np.where(np.isinf(reach), max_reach, reach)

        mm2 = MinMaxScaler()

        reach_scaled = mm2.fit_transform(inf_reach.reshape(-1,1)).ravel()

        self.df['reachability confidence score'] = 1 - reach_scaled

        self.df['reachability severity score'] = reach_scaled

        reach_max = self.df['reachability severity score'].quantile(0.75)
        reach_med = self.df['reachability severity score'].quantile(0.25)

        self.df['reachability severity level'] = np.select([self.df['reachability severity score'] >= reach_max,
                                                             (self.df['reachability severity score'] > reach_med) & (self.df['reachability severity score'] < reach_max)],['High', 'Medium'], default='Low')

        hd = hdbscan.HDBSCAN(min_samples=8, min_cluster_size=7)
        hd.fit(x_pca)
        hd_label = hd.labels_

        self.df['hdbscan label'] = hd_label

        prob = hd.probabilities_.copy()

        self.df['probability confidence score'] = prob

        hd_sever = 1 - self.df['probability confidence score']

        mm5 = MinMaxScaler()
        self.df['probability severity score'] = mm5.fit_transform(hd_sever.to_numpy().reshape(-1,1)).ravel()

        prob_max = self.df['probability severity score'].quantile(0.75)
        prob_med = self.df['probability severity score'].quantile(0.25)

        self.df['probability severity level'] = np.select([(self.df['probability severity score'] >= prob_max),
                                                           (self.df['probability severity score'] > prob_med) & (self.df['probability severity score'] < prob_max)],['High', 'Medium'], default='Low')

        self.df['hdbscan outlier severity score'] = hd.outlier_scores_.copy()

        mm99 = MinMaxScaler()
        self.df['hdbscan outlier severity score'] = mm99.fit_transform(
            self.df['hdbscan outlier severity score'].to_numpy().reshape(-1, 1)
        ).ravel()

        top = self.df['hdbscan outlier severity score'].quantile(0.75)
        middle = self.df['hdbscan outlier severity score'].quantile(0.25)

        self.df['hdbscan outlier severity level'] = np.select([(self.df['hdbscan outlier severity score'] >= top),
                                                               (self.df['hdbscan outlier severity score'] > middle) & (self.df['hdbscan outlier severity score'] < top)],
                                                              ['High', 'Medium'], default='Low')

        lof = LocalOutlierFactor(n_neighbors=100)

        lof_predict = lof.fit_predict(x_pca)

        self.df['local outlier factor label'] = lof_predict

        neg_outlier = lof.negative_outlier_factor_.copy()

        self.df['local outlier factor score'] = neg_outlier

        mm5 = MinMaxScaler()

        local_scale = mm5.fit_transform(neg_outlier.reshape(-1, 1)).ravel()
        self.df['local outlier severity score'] = 1 - local_scale

        local_max = self.df['local outlier severity score'].quantile(0.75)
        local_med = self.df['local outlier severity score'].quantile(0.25)

        self.df['local outlier severity level'] = np.select([self.df['local outlier severity score'] >= local_max,
                                                             (self.df['local outlier severity score'] > local_med) & (self.df['local outlier severity score'] < local_max)],
                                                            ['High', 'Medium'], default='Low')

        density = (self.df['dbscan severity score'] + self.df['reachability severity score'] + self.df['probability severity score'] + self.df['hdbscan outlier severity score'] + self.df['local outlier severity score']) / 5

        mm6 = MinMaxScaler()

        self.df['density severity score'] = mm6.fit_transform(density.to_numpy().reshape(-1,1)).ravel()

        den_top = self.df['density severity score'].quantile(0.75)
        den_mid = self.df['density severity score'].quantile(0.25)

        self.df['density severity level'] = np.select([self.df['density severity score'] >= den_top,
                                                             (self.df['density severity score'] > den_mid) & (
                                                                         self.df['density severity score'] < den_top)],['High', 'Medium'], default='Low')

        iso = IsolationForest(n_estimators=100)

        pre_iso = iso.fit_predict(x_pca)

        self.df['isolation label'] = pre_iso

        iso_decision = iso.decision_function(x_pca)

        mm7 = MinMaxScaler()

        invert_iso = mm7.fit_transform(iso_decision.reshape(-1, 1)).ravel()

        self.df['isolation confidence score'] = invert_iso

        self.df['isolation severity score'] = 1 - invert_iso

        iso_max = self.df['isolation severity score'].quantile(0.75)
        iso_med = self.df['isolation severity score'].quantile(0.25)

        self.df['isolation severity level'] = np.select([self.df['isolation severity score'] >= iso_max,
                                                         (self.df['isolation severity score'] > iso_med) & (self.df['isolation severity score'] < iso_max)],
                                                        ['High', 'Medium'], default='Low')

        density_combine = (self.df['density severity score'] + self.df['isolation severity score']) / 2

        mm8 = MinMaxScaler()

        self.df['final density score'] = mm8.fit_transform(density_combine.to_numpy().reshape(-1,1)).ravel()

        top_d = self.df['final density score'].quantile(0.75)
        mid_d = self.df['final density score'].quantile(0.25)

        self.df['final density level'] = np.select([self.df['final density score'] >= top_d,
                                                    (self.df['final density score'] > mid_d) & (self.df['final density score'] < top_d)],['High', 'Medium'], default='Low')

        svm = OneClassSVM()

        svm_predict = svm.fit_predict(x_pca)

        self.df['svm label'] = svm_predict

        svm_decision = svm.decision_function(x_pca)

        mm9 = MinMaxScaler()
        svm_scale = mm9.fit_transform(svm_decision.reshape(-1,1)).ravel()

        self.df['svm decision confidence score'] = svm_scale

        self.df['svm decision severity score'] = 1 - svm_scale

        top_svm = self.df['svm decision severity score'].quantile(0.75)
        mid_svm = self.df['svm decision severity score'].quantile(0.25)

        self.df['svm severity level'] = np.select([self.df['svm decision severity score'] >= top_svm,
                                                   (self.df['svm decision severity score'] > mid_svm) & (self.df['svm decision severity score']< top_svm)],['High', 'Medium'], default='Low')

        all_combine = (self.df['final density score'] + self.df['svm decision severity score']) / 2
        mm11 = MinMaxScaler()

        self.df['final severity score'] = mm11.fit_transform(all_combine.to_numpy().reshape(-1,1)).ravel()

        top_final = self.df['final severity score'].quantile(0.75)
        mid_final = self.df['final severity score'].quantile(0.25)

        self.df['final severity level'] = np.select([self.df['final severity score'] >= top_final,
                                                     (self.df['final severity score'] > mid_final) & (self.df['final severity score'] < top_final)],['High', 'Medium'], default='Low')


        print(self.df.head(5).to_string())
        self.df = self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)

        #self.df.to_csv('c:/Users/anton/risk/credit_final.csv', index=False)


if __name__ == "__main__":
    a = A()
    a.b()
