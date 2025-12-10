import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

class Gov:
    def __init__(self, file_path, model_name):
        try:
            self.model_name = model_name
            self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python')
            self.df.columns = self.df.columns.str.lower().str.strip()
            #print(self.df.head().to_string())
            #self.df = self.df[~self.df['ledger name'].astype(str).str.contains('Controlled Unclassified|Run on', case=False, na=False)]

            #print(self.df.head().to_string())

            self.cluster = self.df.copy()
            self.copy = self.df.copy()
            self.copy = self.copy.drop(columns=['invoice date', 'terms date', 'hold creation date'])
            #print(self.copy.head().to_string())

            self.num_ber = self.copy.select_dtypes(include=['number', 'complex']).columns.tolist()
            self.obj_ect = self.copy.select_dtypes(include=['category', 'string', 'bool', 'object']).columns.tolist()

            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

            self.num_pipeline = Pipeline([('num_nan', SimpleImputer(strategy='median')),
                                          ('scaler', self.scaler)])

            self.cat_pipeline = Pipeline([('cat_missing', SimpleImputer(strategy='constant', fill_value='missing')),
                                          ('encoder', self.encoder)])

            self.preprocessor = ColumnTransformer([('scaler', self.num_pipeline, self.num_ber),
                                                   ('encoder', self.cat_pipeline, self.obj_ect)])

            self.pipeline = Pipeline([('preprocessor', self.preprocessor),
                                      ('pca', PCA(n_components=4)),
                                      ('dbscan', DBSCAN(eps=0.2, min_samples=5))])

            self.o_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                        ('pca', PCA(n_components=4)),
                                        ('optics', OPTICS(min_samples=5, xi=0.05))])

            self.h_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                        ('pca', PCA(n_components=4)),
                                        ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=5))])

            self.scores = ([silhouette_score, calinski_harabasz_score, davies_bouldin_score])

        except Exception as e: print(f'invalid file path: {e}')

    def p_dbscan(self):
        try:
            self.pipeline.fit(self.copy)

            x = self.pipeline.named_steps['preprocessor'].transform(self.copy)

            x_pca = self.pipeline.named_steps['pca'].transform(x)


            pca_file = pd.DataFrame(x_pca, columns=[f'PC {i+1}' for i in range(x_pca.shape[1])])

            variance = self.pipeline.named_steps['pca'].explained_variance_ratio_
            cumsum = np.cumsum(variance)

            pca_va = [f'PC {i+1}' for i in range(len(variance))]

            var_cum = pd.DataFrame({'PC': pca_va,
                                    'Variance': variance,
                                    'Cumulative': cumsum})

            #print(var_cum)
            y = self.pipeline.named_steps['dbscan'].labels_

            pca_file['dbscan labels'] = y

            pca_file['dbscan identifier'] = (pca_file['dbscan labels'] == -1).astype(int)
            print(pca_file)


            var_cum.insert(0, 'ID', var_cum.index+1)

            var_cum.to_csv('c:/Users/anton/OneDrive/gov_variance_cumsum.csv', index=False)
            print(var_cum.head().to_string())

            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

                s =  0.41270300245472674
                c = 19.749323856315872
                d = 0.6130123983968916

                gov_dbscan = pd.DataFrame({'Silhouette': s,
                                           'Calinski': c,
                                           'Davies': d}, index=[1])
                gov_dbscan.insert(0, 'ID', gov_dbscan.index)
            pca_file.to_csv('c:/Users/anton/OneDrive/gov_pca_dbscan.csv', index=False)
            self.cluster['pca dbscan labels'] = pca_file['dbscan labels']
            self.cluster['pca dbscan identifier'] = pca_file['dbscan identifier']
            self.cluster['pca dbscan category'] = pca_file['dbscan category']
            self.cluster.to_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv', index=False)
            print(self.cluster.head().to_string())

        except Exception as e: print(f'invalid pca - dbscan: {e}')

    def p_optics(self):
        try:
            self.o_pipeline.fit(self.copy)
            x = self.o_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.o_pipeline.named_steps['pca'].transform(x)

            reachability = self.o_pipeline.named_steps['optics'].reachability_
            ordering = self.o_pipeline.named_steps['optics'].ordering_

            reachability = pd.Series(reachability)

            reach = reachability[~np.isinf(reachability)].max()
            reach_df = reachability.replace([np.inf, -np.inf], reach)
            reach_dict = pd.DataFrame({'Reachability': reach_df,
                                      'Ordering': ordering})
            y = self.o_pipeline.named_steps['optics'].labels_
            reach_dict['Optics reachability label'] = y
            reach_dict['Optics reachability identifier'] = (reach_dict['Optics reachability label'] == -1).astype(int)
            reach_dict['Optics reachability category'] = pd.qcut(reach_df, q=4, labels=['critical', 'high', 'medium', 'low'])
            reach_dict.insert(0, "ID", reach_dict.index+1)
            reach_dict.to_csv('c:/Users/anton/OneDrive/gov_optics_reachability.csv', index=False)
            print(reach_dict)



            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.5768397091767461
            c = 62.377261598353755
            d = 0.8067952114345233
            optics_scores = pd.DataFrame({'Silhouette': s,
                                           'Calinski': c,
                                           'Davies': d}, index=[1])
            optics_scores.insert(0, 'ID', optics_scores.index)
            optics_scores.to_csv('c:/Users/anton/OneDrive/gov_optics_scores.csv', index=False)
            print(optics_scores)

            self.cluster1 = pd.read_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv')
            self.cluster1['pca optics labels'] = y
            self.cluster1['pca optics identifier'] = (self.cluster1['pca optics labels'] == -1).astype(int)
            self.cluster1['pca optics category'] = (self.cluster1['pca optics identifier'].apply(lambda x: 'outlier' if x==1 else 'not outlier'))
            self.cluster1.to_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv', index=False)
            print(self.cluster1.head().to_string())
        except Exception as e: print(f'invalid optics: {e}')

    def h_dbscan(self):
        try:
            self.h_pipeline.fit(self.copy)
            x = self.h_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.h_pipeline.named_steps['pca'].transform(x)
            probability = self.h_pipeline.named_steps['hdbscan'].probabilities_
            prob_series = pd.Series(probability)
            prob_cut = pd.cut(prob_series, bins=[-0.1, 0.1, 0.5, 0.7,1.0], labels=['critical', 'high', 'medium', 'low'])
            prob_df = pd.DataFrame({'Probability': prob_series,
                                    'Probability risk': prob_cut})
            y = self.h_pipeline.named_steps['hdbscan'].labels_
            prob_df['hdbscan probability label'] = y
            prob_df['hdbscan probability identifier'] = (prob_df['hdbscan probability label'] == -1).astype(int)
            prob_df['hdbscan probability category'] = prob_df['hdbscan probability identifier'].apply(lambda x: 'outlier' if x==1 else 'not outlier')
            prob_df.insert(0, "ID", prob_df.index+1)
            prob_df.to_csv('c:/Users/anton/OneDrive/gov_hdbscan_probability.csv', index=False)
            print(prob_df)


            self.cluster2 = pd.read_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv')
            self.cluster2['pca hdbscan labels'] = y
            self.cluster2['pca hdbscan identifier'] = (self.cluster2['pca hdbscan labels'] == -1).astype(int)
            self.cluster2['pca hdbscan category'] = (self.cluster2['pca hdbscan identifier'].apply(lambda x: 'outlier' if x==1 else 'not outlier'))
            print(self.cluster2.head().to_string())
            for scores in self.scores:
                s = scores(x_pca, y)
                print(s)
            s = 0.6225862866510431
            c = 63.32913359853615
            d = 0.6678515258522948
            hdbscan_scores = pd.DataFrame({'Silhouette': s,
                                           'Calinski': c,
                                           'Davies': d}, index=[1])
            hdbscan_scores.insert(0, 'ID', hdbscan_scores.index)
            hdbscan_scores.to_csv('c:/Users/anton/OneDrive/gov_hdbscan_score.csv', index=False)

            self.cluster2.to_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv', index=False)
        except Exception as e: print(f'invalid hdbscan: {e}')

    def c_final(self):
        try:
            self.cluster3 = pd.read_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv')

            self.cluster3['total identifier'] = ((self.cluster3['pca dbscan identifier']).astype(int) +
                                                 (self.cluster3['pca optics identifier']).astype(int) +
                                                 (self.cluster3['pca hdbscan identifier']))

            self.cluster3['total category'] = (self.cluster3['total identifier'].map({3: 'critical', 2: 'high', 1: 'medium', 0: 'low'}))

            self.cluster3.insert(0, 'ID', self.cluster3.index +1)
            self.cluster3.to_csv('c:/Users/anton/OneDrive/gov_cluster_final.csv', index=False)

            print(self.cluster3.head().to_string())
        except Exception as e: print(f'invalid final: {e}')


if __name__=='__main__':
    model_name = input('Enter model name here: ')
    g = Gov('c:/Users/anton/OneDrive/invoices_auto.csv', model_name)
    if model_name == 'd':
        g.p_dbscan()
    elif model_name == 'o':
        g.p_optics()
    elif model_name == 'h':
        g.h_dbscan()
    elif model_name == 'f':
        g.c_final()
