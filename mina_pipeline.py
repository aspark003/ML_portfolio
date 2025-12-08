import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class CDbscan:
    def __init__(self, file_path, model_name):
        try:
            self.model_name = model_name
            self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python')
            self.df = self.df.drop(columns=['urgency flag', 'geo distance to vendor', 'invoice match score', 'risk category', 'holiday period', 'is fraud'])
            self.f = self.df.copy()
            self.f.insert(0, 'ID', self.f.index +1)
            #print(self.f.head().to_string())
            self.copy = self.df.copy()

            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.num = self.copy.select_dtypes(include=['number', 'complex']).columns.tolist()
            self.obj = self.copy.select_dtypes(include=['category', 'string', 'object', 'bool']).columns.tolist()
            self.preprocessor = ColumnTransformer([('scaler', self.scaler, self.num),
                                                   ('encoder', self.encoder, self.obj)])
            self.pipeline = Pipeline([('preprocessor', self.preprocessor),
                                      ('pca', PCA(n_components=11)),
                                      ('dbscan', DBSCAN(eps=0.4, min_samples=5))])
            self.optics = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=11)),
                                    ('optics', OPTICS(min_samples=5, xi=0.5))])
            self.hd = Pipeline([('preprocessor', self.preprocessor),
                                ('pca', PCA(n_components=11)),
                                ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=5))])

            self.scores = ([silhouette_score, calinski_harabasz_score, davies_bouldin_score])

        except Exception as e: print(f'invalid file path: {e}')

    def pc_db(self):
        try:
            self.pipeline.fit(self.copy)
            x = self.pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.pipeline.named_steps['pca'].transform(x)

            variance = self.pipeline.named_steps['pca'].explained_variance_ratio_
            cumsum = np.cumsum(variance)
            vr_col = pd.Series([i+1 for i in range(len(variance))])
            vr_pc = pd.Series([i+1 for i in range(x_pca.shape[1])])
            varcum_file = pd.DataFrame({'ID': vr_col,
                                        'PC': vr_pc,
                                        'VARIANCE': variance,
                                        'CUMULATIVE': cumsum})
            varcum_file.to_csv('c:/Users/anton/OneDrive/var_cumsum.csv', index=False)
            #print(varcum_file)

            pca_db_file = pd.DataFrame(x_pca, columns=[f'PC {i + 1}' for i in range(x_pca.shape[1])])

            pca_db_file.insert(0, 'ID', pca_db_file.index +1)
            y = self.pipeline.named_steps['dbscan'].labels_
            pca_db_file['DBSCAN LABELS'] = y
            pca_db_file['DBSCAN IDENTIFIER'] = (pca_db_file['DBSCAN LABELS'] == -1).astype(int)
            pca_db_file['DBSCAN CATEGORY'] = (pca_db_file['DBSCAN IDENTIFIER'].apply(lambda x: 'OUTLIER' if x==1 else 'NOT OUTLIER'))
            pca_db_file.to_csv('c:/Users/anton/OneDrive/pca_db_file.csv', index=False)

            self.f['dbscan labels'] = y
            self.f['dbscan identifier'] = pca_db_file['DBSCAN IDENTIFIER']
            self.f['dbscan category'] = pca_db_file['DBSCAN CATEGORY']
            print(self.f.head().to_string())

            self.f.to_csv('c:/Users/anton/OneDrive/final.csv', index=False)

            #print(pca_db_file)

            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.9062980588772327
            c = 220.6406285038193
            d = 1.0665630128813612

            score_df = pd.DataFrame({'Silhouette score': s,
                                     'Calinski score': c,
                                     'Davies scores': d}, index=[1])
            score_df.insert(0, 'ID', score_df.index)
            score_df.to_csv('c:/Users/anton/OneDrive/cluster_db_scores.csv', index=False)
            #print(score_df)

        except Exception as e: print(f'invalid pca-dbscan: {e}')

    def pc_op(self):
        try:
            self.optics.fit(self.copy)
            x = self.optics.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.optics.named_steps['pca'].transform(x)

            reachability = self.optics.named_steps['optics'].reachability_
            ordering = self.optics.named_steps['optics'].ordering_
            reach = pd.Series(reachability)
            reach_max = reach[~np.isinf(reach)].max()
            reach_replace = reach.replace([np.inf, -np.inf], reach_max)
            reach_df = pd.Series(reach_replace)

            reach_file = pd.DataFrame({'REACHABILITY': reach_df,
                                       'ORDERING': ordering})

            y = self.optics.named_steps['optics'].labels_
            reach_file['OPTICS LABELS'] = y
            reach_file['OPTICS IDENTIFIER'] = (reach_file['OPTICS LABELS'] == -1).astype(int)
            reach_file.insert(0, 'ID', reach_file.index + 1)

            reach_category = pd.qcut(reach_replace, q=4, labels=['low', 'medium', 'high', 'critical'])
            reach_file['RISK CATEGORY'] = reach_category

            self.f = pd.read_csv('c:/Users/anton/OneDrive/final.csv')

            self.f['optics labels'] = reach_file['OPTICS LABELS']
            self.f['optics identifier'] = reach_file['OPTICS IDENTIFIER']
            self.f['optics identifier category'] = (self.f['optics identifier'].apply(lambda x: 'OUTLIER' if x == 1 else 'NOT OUTLIER'))

            self.f['optics risk category'] = reach_file['RISK CATEGORY']
            self.f.to_csv('c:/Users/anton/OneDrive/final.csv', index=False)


            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.905164568415732
            c = 218.57960580240714
            d = 1.0602041995370377

            score_df = pd.DataFrame({'Silhouette score': s,
                                     'Calinski score': c,
                                     'Davies score': d}, index=[1])
            score_df.insert(0, 'ID', score_df.index)


            score_df.to_csv('c:/Users/anton/OneDrive/optics_score.csv', index=False)
            reach_file.to_csv('c:/Users/anton/OneDrive/optics_risk.csv', index=False)

        except Exception as e: print(f'invalid pca - optics: {e}')

    def pc_hd(self):
        try:
            self.hd.fit(self.copy)
            x = self.hd.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.hd.named_steps['pca'].transform(x)
            probability = self.hd.named_steps['hdbscan'].probabilities_
            prob_series = pd.Series(probability)
            prob_col = pd.cut(prob_series, bins=[-0.1, 0.15, 0.30, 0.50, 0.70, 1.00], labels=['critical', 'high', 'medium', 'low', 'stable'])

            prob_df = pd.DataFrame({'PROBABILITY': prob_series})
            y = self.hd.named_steps['hdbscan'].labels_
            prob_df['HDBSCAN LABELS'] = y
            prob_df['HDBSCAN IDENTIFIER'] = (prob_df['HDBSCAN LABELS'] == -1).astype(int)
            prob_df.insert(0, 'ID', prob_df.index + 1)
            prob_df['HDBSCAN RISK CATEGORY'] = prob_col
            #print(prob_df.head().to_string())

            self.f = pd.read_csv('c:/Users/anton/OneDrive/final.csv')

            self.f['hdbscan labels'] = prob_df['HDBSCAN LABELS']
            self.f['hdbscan identifier'] = prob_df['HDBSCAN IDENTIFIER']
            self.f['hdbscan risk identifier'] = (self.f['hdbscan identifier'].apply(lambda x: 'OUTLIER' if x== 1 else 'NOT OUTLIER'))
            self.f['hdbscan risk category'] = prob_df['HDBSCAN RISK CATEGORY']

            #print(self.f.head().to_string())
            self.f.to_csv('c:/Users/anton/OneDrive/final.csv', index=False)
            print(self.f.head().to_string())

            prob_df.to_csv('c:/Users/anton/OneDrive/hd_probability.csv', index=False)
            for scores in self.scores:
                s = scores(x_pca,y)
                print(f'{scores.__name__}: {s}')

            s = 0.9091081070645468
            c = 349.834424864566
            d = 1.0780923398047277

            hd_scores = pd.DataFrame({'Silhouette score': s,
                                      'Calinski score': c,
                                      'Davies score': d}, index=[1])
            hd_scores.insert(0, 'ID', hd_scores.index)
            hd_scores.to_csv('c:/Users/anton/OneDrive/hd_scores.csv', index=False)

        except Exception as e: print(f'invalid hdbscan: {e}')

    def final_db(self):
        try:
            self.f = pd.read_csv('c:/Users/anton/OneDrive/final.csv')
            self.f['total identifier'] = ((self.f['dbscan identifier'] == 1 ).astype(int)+
                          (self.f['optics identifier'] == 1).astype(int)+
                          (self.f['hdbscan identifier'] == 1).astype(int))
            self.f['total category risk'] = self.f['total identifier'].map({0: 'low', 1: 'medium', 2: 'high', 3: 'critical'})
            self.f.to_csv('c:/Users/anton/OneDrive/final.csv', index=False)
            print(self.f.head().to_string())
        except Exception as e: print(f'invalid final: {e}')
if __name__ == "__main__":
    model_name = input('Enter model name here: ')
    cd = CDbscan('c:/Users/anton/OneDrive/park_consultant.csv', model_name)
    if model_name == 'd':
        cd.pc_db()
    elif model_name == 'o':
        cd.pc_op()
    elif model_name == 'h':
        cd.pc_hd()
    elif model_name == 'f':
        cd.final_db()
