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
            #print(self.cluster.head().to_string())
            self.pipeline.fit(self.copy)
            x = self.pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.pipeline.named_steps['pca'].transform(x)
            pca_file = pd.DataFrame(x_pca, columns=[f'PC {i+1}' for i in range(x_pca.shape[1])])

            y = self.pipeline.named_steps['dbscan'].labels_

            pca_file['pca-dbscan labels'] = y
            pca_file['pca-dbscan identifier'] = (pca_file['pca-dbscan labels'] == -1).astype(int)
            pca_file['pca-dbscan category'] = (pca_file['pca-dbscan identifier'].apply(lambda x: 'outlier' if x==1 else 'not outlier'))

            pca_file.insert(0, 'ID', pca_file.index+1)

            pca_file.to_csv('c:/Users/anton/OneDrive/pca_file.csv', index=False)


            print(pca_file)

            var = self.pipeline.named_steps['pca'].explained_variance_ratio_

            cum = np.cumsum(var)

            var_col = pd.Series([f'PC {i+1}' for i in range(len(var))])
            var_cum_df = pd.DataFrame({'PC': var_col,
                                       'Variance': var,
                                       'Cumulative': cum})
            var_cum_df.insert(0,'ID', var_cum_df.index + 1)
            var_cum_df.to_csv('c:/Users/anton/OneDrive/variance_cumulative.csv', index=False)


            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.41270300245472674
            c = 19.749323856315872
            d = 0.6130123983968916

            dbscan_scores = pd.DataFrame({'DBSCAN Silhouette scores': s,
                                          'DBSCAN Calinski scores': c,
                                          'DBSCAN Davies scores': d}, index=[1])
            dbscan_scores.insert(0, 'ID', dbscan_scores.index)
            dbscan_scores.to_csv('c:/Users/anton/OneDrive/p_dbscan_scores.csv', index=False)
            #print(dbscan_scores)
            self.cluster['dbscan labels'] = y
            self.cluster['dbscan identifier'] = (self.cluster['dbscan labels'] == -1).astype(int)
            self.cluster['dbscan outlier category'] = (self.cluster['dbscan identifier'].apply(lambda x: 'outlier' if x == 1 else 'not outlier'))
            self.cluster.to_csv('c:/Users/anton/OneDrive/dbscan_outliers.csv', index=False)
            self.cluster.insert(0, 'ID', self.cluster.index+1)
            self.cluster.to_csv('c:/Users/anton/OneDrive/dbscan_outliers.csv', index=False)
            #print(self.cluster.head().to_string())

        except Exception as e: print(f'invalid pca - dbscan: {e}')

    def p_optics(self):
        try:
            self.cluster_optics = pd.read_csv('c:/Users/anton/OneDrive/dbscan_outliers.csv')
            #print(self.cluster_optics.head().to_string())
            self.o_pipeline.fit(self.copy)
            x = self.o_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.o_pipeline.named_steps['pca'].transform(x)
            reachability = self.o_pipeline.named_steps['optics'].reachability_
            ordering = self.o_pipeline.named_steps['optics'].ordering_
            reach = pd.Series(reachability)

            reach_max = reach[~np.isinf(reach)].max()
            replace_max = reach.replace([np.inf, -np.inf], reach_max)
            reachability_df = pd.Series(replace_max)
            optics_reach = pd.DataFrame({'Reachability': reachability_df,
                                         'Ordering': ordering})

            y = self.o_pipeline.named_steps['optics'].labels_
            optics_reach['Reachability labels'] = y
            self.cluster_optics['Reachability labels'] = y
            optics_reach['Reachability identifier'] = (optics_reach['Reachability labels'] == -1).astype(int)
            self.cluster_optics['Reachability identifier'] = optics_reach['Reachability identifier']
            optics_reach.insert(0, 'ID', optics_reach.index+1)


            reach_category = pd.qcut(optics_reach['Reachability'], q=4, labels=['low', 'medium', 'high', 'critical'])

            optics_reach['Reachability risk'] = reach_category
            self.cluster_optics['Reachability risk'] = reach_category

            optics_reach.to_csv('c:/Users/anton/OneDrive/reachability_ordering.csv', index=False)

            self.cluster_optics.to_csv('c:/Users/anton/OneDrive/gov_final.csv', index=False)

            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.5768397091767461
            c = 62.377261598353755
            d = 0.8067952114345233

            reach_file = pd.DataFrame({'Silhouette score': s,
                                       'Calinski score': c,
                                       'Davies scores': d}, index=[1])
            reach_file.insert(0, 'ID', reach_file.index)
            reach_file.to_csv('c:/Users/anton/OneDrive/optics_reach_scores.csv', index=False)
            #print(reach_file)
        except Exception as e: print(f'invalid pca-optics: {e}')

    def h_dbscan(self):
        try:
            self.cluster_optics = pd.read_csv('c:/Users/anton/OneDrive/gov_final.csv')
            #print(self.cluster_optics.head().to_string())
            self.h_pipeline.fit(self.copy)
            x = self.h_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.h_pipeline.named_steps['pca'].transform(x)

            probability = self.h_pipeline.named_steps['hdbscan'].probabilities_
            hd_prob = pd.Series(probability)
            hd_prob_df = pd.cut(hd_prob, bins=[-0.1,0.15,0.50,0.70,1.00], labels=['critical', 'high', 'medium', 'low'])
            pca_hd = pd.DataFrame({'Probability': hd_prob})
            y = self.h_pipeline.named_steps['hdbscan'].labels_
            pca_hd['hdbscan labels'] = y
            self.cluster_optics['HDBSCAN labels'] = y
            pca_hd['hdbscan identifiers'] = (pca_hd['hdbscan labels'] == -1).astype(int)
            self.cluster_optics['HDBSCAN identifiers'] = pca_hd['hdbscan identifiers']
            pca_hd['hdbscan probability category'] = hd_prob_df
            self.cluster_optics['HDBSCAN probabilility category'] = pca_hd['hdbscan probability category']
            pca_hd.insert(0, 'ID', pca_hd.index+1)
            pca_hd.to_csv('c:/Users/anton/OneDrive/hdbscan_prob_category.csv', index=False)

            self.cluster_optics.to_csv('c:/Users/anton/OneDrive/gov_hdbscan.csv', index=False)
            for scores in self.scores:
                s = scores(x_pca, y)
                print(f'{scores.__name__}: {s}')

            s = 0.6225862866510431
            c = 63.32913359853615
            d = 0.6678515258522948

            hd_prob_scores = pd.DataFrame({'Silhouette scores': s,
                                           'Calinski scores': c,
                                           'Davies scores': d}, index=[1])
            hd_prob_scores.insert(0, 'ID', hd_prob_scores.index)
            hd_prob_scores.to_csv('c:/Users/anton/OneDrive/gov_hd_probability_scores.csv', index=False)
            print(hd_prob_scores)
        except Exception as e: print(f'invalid hdbscan: {e}')

    def gov_final(self):
        try:
            final_gov = pd.read_csv('c:/Users/anton/OneDrive/gov_hdbscan.csv')
            final_gov['Total identifier'] = pd.DataFrame((final_gov['dbscan identifier']).astype(int)+
                                                         (final_gov['Reachability identifier']).astype(int)+
                                                         (final_gov['HDBSCAN identifiers']).astype(int))

            final_gov['Total category'] = final_gov['Total identifier'].map({0: 'stable', 1: 'low', 2: 'medium', 3: 'critical'})
            final_gov.to_csv('c:/Users/anton/OneDrive/gov_total.csv', index=False)
            print(final_gov.head().to_string())
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
        g.gov_final()
