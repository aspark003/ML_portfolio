import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from yellowbrick.features import PCA as PCAVisualizer
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

class FinalFraud:
    def __init__(self, file_path, model_name):
        try:
            self.model_name = model_name
            self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python')
            self.df = self.df.drop(columns=['risk category', 'is fraud', 'urgency flag', 'holiday period'])
            self.pca_copy = self.df.copy()
            self.copy = self.df.copy()

            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.num = self.copy.select_dtypes(include=['number', 'complex']).columns.tolist()
            self.obj = self.copy.select_dtypes(include=['object', 'string', 'bool', 'category']).columns.tolist()
            self.preprocessor = ColumnTransformer([('scaler', self.scaler, self.num),
                                                   ('encoder', self.encoder, self.obj)])
            self.dbscan = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=12)),
                                    ('dbscan', DBSCAN(eps=0.5, min_samples=3))])
            self.optics = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=12)),
                                    ('optics', OPTICS(min_samples=5, xi=0.07))])
            self.hdbscan = Pipeline([('preprocessor', self.preprocessor),
                                     ('pca', PCA(n_components=12)),
                                     ('hdbscan', HDBSCAN(min_samples=3, min_cluster_size=3))])
            self.scores = ([silhouette_score, calinski_harabasz_score, davies_bouldin_score])

        except Exception as e: print(f'invalid file path: {e}')

    def db_scan(self):
        try:
            self.dbscan.fit(self.copy)
            x = self.dbscan.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.dbscan.named_steps['pca'].transform(x)
            variance = self.dbscan.named_steps['pca'].explained_variance_ratio_
            c_variance = np.cumsum(variance)

            pc_col = ([f'PC {i + 1}' for i in range(len(variance))])

            cum_var_file = pd.DataFrame({'PC': pc_col,
                                         'VARIANCE' : variance,
                                         'CUMULATIVE' : c_variance})

            cum_var_file.to_csv('c:/Users/anton/OneDrive/cumsum_variance.csv', index=False)

            y = self.dbscan.named_steps['dbscan'].labels_
            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            self.pca_copy['dbscan label'] = y
            self.pca_copy['dbscan fraud label'] = (self.pca_copy['dbscan label'] == -1).astype(int)
            db_fraud = self.pca_copy['dbscan fraud label'].apply(lambda x: 'fraud' if x == 1 else 'not fraud')

            self.pca_copy['dbscan fraud category'] = db_fraud

            print(self.pca_copy.head().to_string())

            self.pca_copy.to_csv('c:/Users/anton/OneDrive/dbscan_label.csv', index=False)

            ss = 0.94864612848988
            chs = 2376.771263564452
            dbs = 0.9306725583814839

            dbscan_scores = pd.DataFrame({'dbscan silhouette': ss,
                                          'dbscan calinski': chs,
                                          'dbscan davies': dbs,}, index=[0])

            dbscan_scores.to_csv('c:/Users/anton/OneDrive/dbscan_scores.csv', index=False)

            print(dbscan_scores)
        except Exception as e: print(f'invalid dbscan scores: {e}')

    def op_tics(self):
        try:
            optic_scores = pd.read_csv('c:/Users/anton/OneDrive/dbscan_scores.csv')
            optic_labels = pd.read_csv('c:/Users/anton/OneDrive/dbscan_label.csv')

            self.optics.fit(self.copy)
            x = self.optics.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.optics.named_steps['pca'].transform(x)
            y = self.optics.named_steps['optics'].labels_

            optic_labels['optics label'] = y
            optic_labels['optics fraud label'] = (optic_labels['optics label'] == -1).astype(int)

            order = self.optics.named_steps['optics'].ordering_
            reach = self.optics.named_steps['optics'].reachability_
            reach_series = pd.Series(reach)
            reach_max = reach_series[~np.isinf(reach_series)].max()
            reach_replace = reach_series.replace([np.inf, -np.inf], reach_max)

            reach_risk_cat = pd.qcut(reach_replace, q=4, labels=['low', 'medium', 'high', 'critical'])

            reach_file = pd.DataFrame({'reachability': reach_replace,
                                       'ordering': order,
                                       'risk category': reach_risk_cat})
            reach_file.to_csv('c:/Users/anton/OneDrive/optics_risk.csv', index=False)

            optic_labels['optics fraud category'] = optic_labels['optics fraud label'].apply(lambda x: 'fraud' if x == 1 else 'not fraud')

            optic_labels['optics reachability category'] = reach_risk_cat
            print(optic_labels.head().to_string())
            print(reach_file.head().to_string())
            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}:{s}')

            optic_labels.to_csv('c:/Users/anton/OneDrive/optic_reach_scores.csv', index=False)
            ss = 0.8670729481702407
            chs = 156.0125369478855
            dbs = 1.0455596451663733

            o_score = pd.DataFrame({'silhouette': ss,
                                        'calinski': chs,
                                        'davis': dbs}, index=[0])
            o_score.to_csv('c:/Users/anton/OneDrive/optic_scores.csv', index=False)

        except Exception as e: print(f'invalid optics score: {e}')

    def hd_bscan(self):
        try:
            self.hdbscan.fit(self.copy)
            x = self.hdbscan.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.hdbscan.named_steps['pca'].transform(x)
            y = self.hdbscan.named_steps['hdbscan'].labels_
            probability = self.hdbscan.named_steps['hdbscan'].probabilities_
            prob_file = pd.Series(probability)
            hd_prob = pd.cut(prob_file, bins=[-0.1, 0.1, 0.5, 0.7, 1.0],
                             labels=['critical', 'high', 'medium', 'low'])

            prob_risk = pd.DataFrame({'probability': prob_file,
                                      'probability risk': hd_prob})
            prob_risk.to_csv('c:/Users/anton/OneDrive/hdbscan_prob.csv', index=False)

            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            ss = 0.9479455467541876
            chs = 3065.743513701147
            dbs = 0.9124938715546878
            hdbscan_scores = pd.DataFrame({'hd silhouette': ss,
                                           'hd calinski': chs,
                                           'hd davies': dbs}, index=[0])
            hdbscan_scores.to_csv('c:/Users/anton/OneDrive/hdbscan_score.csv', index=False)
            print(hdbscan_scores)

            hd = pd.read_csv('c:/Users/anton/OneDrive/optic_reach_scores.csv')
            hd['hdbscan label'] = y
            hd['hdbscan fraud label'] = (hd['hdbscan label'] == -1).astype(int)
            hd['hdbscan fraud category'] = (hd['hdbscan fraud label'].apply(lambda x: 'fraud' if x == 1 else 'not fraud'))
            hd.to_csv('c:/Users/anton/OneDrive/hdbscan_join.csv', index=False)
            hd_join = pd.read_csv('c:/Users/anton/OneDrive/hdbscan_join.csv')
            hdbscan_file = hd_join.join(prob_risk)
            hdbscan_file.to_csv('c:/Users/anton/OneDrive/hdbscan_prob_final.csv', index=False)
            print(hdbscan_file.head().to_string())
        except Exception as e: print(f'invalid hdbscan: {e}')

    def all_model(self):
        try:
            all = pd.read_csv('c:/Users/anton/OneDrive/hdbscan_prob_final.csv')

            all['total fraud label'] = ((all['dbscan fraud label'] == 1).astype(int)+
                          (all['optics fraud label'] == 1).astype(int)+
                          (all['hdbscan fraud label'] == 1).astype(int))

            all['final fraud category'] = all['total fraud label'].map({0:'low',1:'medium',2:'high',3:'critical' })
            all.to_csv('c:/Users/anton/OneDrive/final_fraud.csv', index=False)

            db = pd.read_csv('c:/Users/anton/OneDrive/dbscan_scores.csv')
            o = pd.read_csv('c:/Users/anton/OneDrive/optic_scores.csv')
            hd = pd.read_csv('c:/Users/anton/OneDrive/hdbscan_score.csv')
            #print(o.head().to_string())
            db = db.rename(columns=({'silhouette': 'dbscan silhouette',
                                     'calinski': 'dbscan calinski',
                                     'davies': 'dbscan davies'}))

            o = o.rename(columns=({'silhouette': 'optics silhouette',
                                   'calinski': 'optics calinski',
                                   'davis': 'optics davies'}))

            hd = hd.rename(columns=({'hd silhouette': 'hdbscan silhouette',
                                     'hd calinski': 'hdbscan calinski',
                                     'hd davies': 'hdbscan davies'}))

            final_fraud_scores = db.join(o).join(hd)
            final_fraud_scores.to_csv('c:/Users/anton/OneDrive/cluster_fraud.csv', index=False)

            print(final_fraud_scores.head().to_string())




            #print(all.head().to_string())
        except Exception as e: print(f'invalid all models: {e}')

if __name__ == "__main__":
    model_name = input('Enter model name here: ')
    ff = FinalFraud('c:/Users/anton/OneDrive/park_consultant.csv', model_name)
    if model_name == 'd':
        ff.db_scan()
    elif model_name == 'o':
        ff.op_tics()
    elif model_name == 'h':
        ff.hd_bscan()
    elif model_name == 'a':
        ff.all_model()
