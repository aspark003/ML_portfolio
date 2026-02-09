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
                                               ('o', obj_pipeline, self.obj)],
                                              remainder='drop',
                                              sparse_threshold=0)

        # Do not overwrite self.copy with the transformed numpy array
        # Keep transformed matrix in X for downstream modeling
        self.X = self.preprocessor.fit_transform(self.copy)

        self.scores = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]

    def b(self):

        # PCA on transformed features
        pca = PCA(n_components=0.9, svd_solver='auto', random_state=42)
        x_pca = pca.fit_transform(self.X)

        # DBSCAN on PCA space
        dbscan = DBSCAN(eps=0.2, min_samples=7, metric='euclidean', n_jobs=-1)
        dbscan.fit(x_pca)
        db_label = dbscan.labels_

        # attach DBSCAN labels to original dataframe copy
        self.df['dbscan label'] = db_label

        # ---- FIX: do not treat noise (-1) like a normal "cluster size"
        # Cluster sizes excluding noise
        vc = self.df['dbscan label'].value_counts(dropna=False)
        self.df['dbscan label size'] = self.df['dbscan label'].map(vc).fillna(0).astype(int)

        # Size-based "confidence" for non-noise points only
        non_noise_mask = (self.df['dbscan label'] != -1)

        self.df['dbscan confidence score'] = 0.0
        if non_noise_mask.any():
            mm1 = MinMaxScaler()
            self.df.loc[non_noise_mask, 'dbscan confidence score'] = mm1.fit_transform(
                self.df.loc[non_noise_mask, 'dbscan label size'].to_numpy().reshape(-1, 1)
            ).ravel()

        # Severity: noise points are highest severity; otherwise inverse of confidence
        self.df['dbscan severity score'] = np.where(
            non_noise_mask,
            1.0 - self.df['dbscan confidence score'].to_numpy(),
            1.0
        )

        db_top = self.df['dbscan severity score'].quantile(0.75)
        db_mid = self.df['dbscan severity score'].quantile(0.25)

        self.df['dbscan severity level'] = np.select(
            [self.df['dbscan severity score'] >= db_top,
             (self.df['dbscan severity score'] > db_mid) & (self.df['dbscan severity score'] < db_top)],
            ['High', 'Medium'],
            default='Low'
        )

        # OPTICS on PCA space
        op = OPTICS(min_samples=3, xi=0.05, metric='euclidean')
        op.fit(x_pca)
        op_tics = op.labels_
        self.df['optics label'] = op_tics

        reach = op.reachability_.copy()

        finite_mask = np.isfinite(reach)
        if finite_mask.any():
            max_reach = np.nanmax(reach[finite_mask])
            reach_replace = np.where(finite_mask, reach, max_reach)
        else:
            reach_replace = np.nan_to_num(reach, nan=0.0, posinf=0.0, neginf=0.0)

        # If any NaNs remain, replace with median
        if np.isnan(reach_replace).any():
            reach_replace = np.nan_to_num(reach_replace, nan=np.nanmedian(reach_replace))

        self.df['reachability'] = reach_replace

        mm2 = MinMaxScaler()
        reach_scaled = mm2.fit_transform(reach_replace.reshape(-1, 1)).ravel()
        self.df['optics reachability severity score'] = reach_scaled

        reach_max = self.df['optics reachability severity score'].quantile(0.75)
        reach_med = self.df['optics reachability severity score'].quantile(0.25)

        self.df['optics reachability severity level'] = np.select(
            [self.df['optics reachability severity score'] >= reach_max,
             (self.df['optics reachability severity score'] > reach_med) & (self.df['optics reachability severity score'] < reach_max)],
            ['High', 'Medium'],
            default='Low')


        hd = hdbscan.HDBSCAN(min_samples=8, min_cluster_size=7, cluster_selection_method='eom', metric='euclidean')
        hd.fit(x_pca)
        hd_label = hd.labels_
        self.df['hdbscan label'] = hd_label

        outlier_scores = getattr(hd, 'outlier_scores_', None)
        if outlier_scores is None:
            prob = getattr(hd, 'probabilities_', None)
            if prob is not None:
                outlier_scores = 1.0 - prob
            else:
                outlier_scores = np.zeros(len(self.df))

        outlier_scores = np.asarray(outlier_scores, dtype=float)
        if np.all(np.isfinite(outlier_scores)) and (np.nanmax(outlier_scores) - np.nanmin(outlier_scores) > 0):
            mm_h = MinMaxScaler()
            outlier_scores_scaled = mm_h.fit_transform(outlier_scores.reshape(-1, 1)).ravel()
        else:
            outlier_scores_scaled = np.zeros(len(self.df), dtype=float)

        self.df['hdbscan outlier severity score'] = outlier_scores_scaled

        top = self.df['hdbscan outlier severity score'].quantile(0.75)
        middle = self.df['hdbscan outlier severity score'].quantile(0.25)

        self.df['hdbscan outlier severity level'] = np.select(
            [(self.df['hdbscan outlier severity score'] >= top),
             (self.df['hdbscan outlier severity score'] > middle) & (self.df['hdbscan outlier severity score'] < top)],
            ['High', 'Medium'],
            default='Low')

        anomaly_scale = (
            self.df['dbscan severity score'].fillna(0.0)
            + self.df['optics reachability severity score'].fillna(0.0)
            + self.df['hdbscan outlier severity score'].fillna(0.0)) / 3.0

        mm22 = MinMaxScaler()
        self.df['density anomaly score'] = mm22.fit_transform(anomaly_scale.to_numpy().reshape(-1, 1)).ravel()

        db_op_hd_critical = self.df['density anomaly score'].quantile(0.95)
        db_op_hd_max = self.df['density anomaly score'].quantile(0.85)
        db_op_hd_med = self.df['density anomaly score'].quantile(0.60)

        self.df['density severity level'] = np.select(
            [self.df['density anomaly score'] >= db_op_hd_critical,
             self.df['density anomaly score'] >= db_op_hd_max,
             self.df['density anomaly score'] >= db_op_hd_med],
            ['Critical', 'High', 'Medium'],
            default='Low')

        n_samples = x_pca.shape[0]
        n_neighbors = 100
        if n_samples <= 2:
            n_neighbors = 1
        else:
            n_neighbors = min(n_neighbors, n_samples - 1)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='euclidean', novelty=False)
        lof_predict = lof.fit_predict(x_pca)
        self.df['local outlier factor label'] = lof_predict

        neg_outlier = lof.negative_outlier_factor_.copy()
        scaled_neg_outlier = -neg_outlier
        mm5 = MinMaxScaler()
        self.df['local outlier severity score'] = mm5.fit_transform(scaled_neg_outlier.reshape(-1, 1)).ravel()

        local_max = self.df['local outlier severity score'].quantile(0.75)
        local_med = self.df['local outlier severity score'].quantile(0.25)

        self.df['local outlier severity level'] = np.select(
            [self.df['local outlier severity score'] >= local_max,
             (self.df['local outlier severity score'] > local_med) & (self.df['local outlier severity score'] < local_max)],
            ['High', 'Medium'],default='Low')


        iso = IsolationForest(n_estimators=100, max_samples='auto', random_state=42)
        pre_iso = iso.fit_predict(x_pca)
        self.df['isolation label'] = pre_iso

        iso_decision = iso.decision_function(x_pca)
        invert_iso = -iso_decision
        mm7 = MinMaxScaler()
        self.df['isolation severity score'] = mm7.fit_transform(invert_iso.reshape(-1, 1)).ravel()

        iso_max = self.df['isolation severity score'].quantile(0.75)
        iso_med = self.df['isolation severity score'].quantile(0.25)

        self.df['isolation severity level'] = np.select(
            [self.df['isolation severity score'] >= iso_max,
             (self.df['isolation severity score'] > iso_med) & (self.df['isolation severity score'] < iso_max)],
            ['High', 'Medium'],default='Low')


        svm = OneClassSVM(kernel='rbf', gamma='scale')
        svm_predict = svm.fit_predict(x_pca)
        self.df['svm label'] = svm_predict

        decision_svm = svm.decision_function(x_pca)
        invert_decision = -decision_svm
        mm9 = MinMaxScaler()
        self.df['svm severity score'] = mm9.fit_transform(invert_decision.reshape(-1, 1)).ravel()

        top_svm = self.df['svm severity score'].quantile(0.75)
        mid_svm = self.df['svm severity score'].quantile(0.25)

        self.df['svm severity level'] = np.select(
            [self.df['svm severity score'] >= top_svm,
             (self.df['svm severity score'] > mid_svm) & (self.df['svm severity score'] < top_svm)],
            ['High', 'Medium'],default='Low')

        decision_scores = (
            self.df['local outlier severity score'].fillna(0.0)
            + self.df['isolation severity score'].fillna(0.0)
            + self.df['svm severity score'].fillna(0.0)) / 3.0

        mm11 = MinMaxScaler()
        self.df['decision severity score'] = mm11.fit_transform(decision_scores.to_numpy().reshape(-1, 1)).ravel()

        critical_final = self.df['decision severity score'].quantile(0.95)
        top_final = self.df['decision severity score'].quantile(0.85)
        mid_final = self.df['decision severity score'].quantile(0.60)

        self.df['decision severity level'] = np.select(
            [self.df['decision severity score'] >= critical_final,
             self.df['decision severity score'] >= top_final,
             self.df['decision severity score'] >= mid_final],
            ['Critical', 'High', 'Medium'],default='Low')

        detection_score = (
            self.df['density anomaly score'].fillna(0.0)
            + self.df['decision severity score'].fillna(0.0)) / 2.0

        mm15 = MinMaxScaler()
        self.df['risk detection score'] = mm15.fit_transform(detection_score.to_numpy().reshape(-1, 1)).ravel()

        risk_critical = self.df['risk detection score'].quantile(0.95)
        risk_max = self.df['risk detection score'].quantile(0.85)
        risk_med = self.df['risk detection score'].quantile(0.60)

        self.df['risk detection level'] = np.select(
            [self.df['risk detection score'] >= risk_critical,
             self.df['risk detection score'] >= risk_max,
             self.df['risk detection score'] >= risk_med],
            ['Critical', 'High', 'Medium'],default='Low')

        self.df = self.df.reset_index(drop=True)
        self.df.insert(0, 'id', self.df.index + 1)
        #print(self.df.head(3).to_string())

        # Uncomment to write results
        # self.df.to_csv('c:/Users/anton/risk/credit_final.csv', index=False)


if __name__ == "__main__":
    a = A()
    a.b()
