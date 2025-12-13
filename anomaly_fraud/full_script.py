import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ClusterF:
    def __init__(self, file_path, model_name):
        try:
            self.model_name = model_name
            self.df = pd.read_csv(file_path)

            self.copy = self.df.copy()
            self.combine = self.copy.copy()
            self.num = self.copy.select_dtypes(include=[np.number]).columns.tolist()
            self.obj = self.copy.select_dtypes(include=['object', 'string', 'bool', 'category']).columns.tolist()

            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

            self.num_imp = Pipeline([('n_imp', SimpleImputer(strategy='median')),
                                         ('scale', self.scaler)])

            self.obj_imp = Pipeline([('o_imp', SimpleImputer(strategy='constant', fill_value='missing')),
                                         ('encode', self.encoder)])

            self.preprocessor = ColumnTransformer([('scaler', self.num_imp, self.num),
                                                   ('encoder', self.obj_imp, self.obj)])

            self.db_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                      ('pca', PCA(n_components=3)),
                                      ('dbscan', DBSCAN(eps=0.6, min_samples=5))])

            self.iso_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                      ('pca', PCA(n_components=3)),
                                      ('iso', IsolationForest(contamination=0.2, random_state=42))])

            self.op_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                         ('pca', PCA(n_components=4)),
                                         ('optics', OPTICS(min_samples=4, xi=0.05))])

            self.iso_optics_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                        ('pca', PCA(n_components=4)),
                                        ('iso', IsolationForest(contamination=0.2, random_state=42))])

            self.hd_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                         ('pca', PCA(n_components=4)),
                                         ('hd', HDBSCAN(min_samples=4, min_cluster_size=4))])

            self.hd_iso_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                             ('pca', PCA(n_components=4)),
                                             ('iso', IsolationForest(contamination=0.2, random_state=42))])

            self.scores = ([silhouette_score, calinski_harabasz_score, davies_bouldin_score])

            #print(self.copy.head().to_string())

        except Exception as e: print(f'invalid file path: {e}')


    def db_scan(self):
        try:
            self.db_pipeline.fit(self.copy)
            x = self.db_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.db_pipeline.named_steps['pca'].transform(x)

            variance = self.db_pipeline.named_steps['pca'].explained_variance_ratio_
            #print(variance)

            cumsum = np.cumsum(variance)
            #print(cumsum)

            y = self.db_pipeline.named_steps['dbscan'].labels_

            for scores in self.scores:
                s = scores(x_pca, y)
                #print(f'{scores.__name__}:{s}')
            #print(x_pca)
            #print(set(y))

            self.combine['dbscan labels'] = y
            self.combine['dbscan identifier'] = (self.combine['dbscan labels'] == -1).astype(int)
            self.combine['dbscan category'] = self.combine['dbscan identifier'].apply(lambda x: 'Anomaly' if x == 1 else 'Not anomaly')

            #Isolation initiated
            self.iso_pipeline.fit(self.copy)

            x_iso = self.iso_pipeline.named_steps['preprocessor'].transform(self.copy)
            pca_iso = self.iso_pipeline.named_steps['pca'].transform(x_iso)

            y_iso = self.iso_pipeline.named_steps['iso'].fit(pca_iso)

            iso_predict = self.iso_pipeline.named_steps['iso'].predict(pca_iso)
            iso_scores = self.iso_pipeline.named_steps['iso'].decision_function(pca_iso)
            print(iso_scores)
            self.combine['dbscan isolation labels'] = iso_predict
            self.combine['dbscan isolation anomaly identifier'] = (self.combine['dbscan isolation labels'] == -1).astype(int)
            self.combine['dbscan isolation anomaly category'] = self.combine['dbscan isolation anomaly identifier'].apply(lambda x: 'Anomaly' if x == 1 else 'Not anomaly')
            iso_plot = pd.DataFrame({'dbscan isolation scores' : iso_scores,
                                     'dbscan isolation anomaly identifier': iso_predict})

            iso_plot['dbscan isolation anomaly category'] = iso_plot['dbscan isolation anomaly identifier'].apply(lambda x: 'Anomaly' if x == 1 else 'not anomaly')
            iso_plot.insert(0, 'id', iso_plot.index +1)
            iso_plot.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_dbscan_iso.csv', index=False)


            #print(self.combine.head().to_string())


            self.combine.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file1.csv', index=False)

            print(self.combine.head().to_string())
        except Exception as e:print(f'invalid dbscan scores: {e}')

    def op_tics(self):
        try:
            self.combine2 = pd.read_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file1.csv')

            self.op_pipeline.fit(self.copy)
            x = self.op_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.op_pipeline.named_steps['pca'].transform(x)
            y = self.op_pipeline.named_steps['optics'].labels_
            for scores in self.scores:
                s = scores(x_pca, y)
                #print(f'{scores.__name__}: {s}')

            self.combine2['optics labels'] = y
            self.combine2['optics identifier'] = (self.combine2['optics labels'] == -1).astype(int)
            self.combine2['optics category'] = self.combine2['optics identifier'].apply(lambda x: 'Anomaly' if x == 1 else 'Not anomaly')

            self.iso_optics_pipeline.fit(self.copy)
            x_iso = self.iso_optics_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_iso_pca = self.iso_optics_pipeline.named_steps['pca'].transform(x_iso)

            y_iso_fit = self.iso_optics_pipeline.named_steps['iso'].fit(x_iso_pca)
            y_iso_predict = self.iso_optics_pipeline.named_steps['iso'].predict(x_iso_pca)
            y_iso_decision = self.iso_optics_pipeline.named_steps['iso'].decision_function(x_iso_pca)

            optics_iso_scores = pd.DataFrame({'Optics isolation scores': y_iso_decision,
                                              'Optics anomaly detector': y_iso_predict})
            optics_iso_scores.insert(0, 'id', optics_iso_scores.index+1)
            optics_iso_scores.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_optics_iso.csv', index=False)
            #print(optics_iso_scores)
            #print(self.combine2.head().to_string())

            self.combine2['optics isolation labels'] = y_iso_predict
            self.combine2['optics isolation anomaly identifier'] = (self.combine2['optics isolation labels'] == -1).astype(int)
            self.combine2['optics isolation anomaly category'] = self.combine2['optics isolation anomaly identifier'].apply(lambda x: 'Anomaly' if x==1 else 'Not anomaly')
            self.combine2.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file2.csv', index=False)
            print(self.combine2.head().to_string())
        except Exception as e: print(f'invalid optics: {e}')


    def hd_scan(self):
        try:
            self.combine3 = pd.read_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file2.csv')
            self.hd_pipeline.fit(self.copy)
            x = self.hd_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_pca = self.hd_pipeline.named_steps['pca'].transform(x)
            y = self.hd_pipeline.named_steps['hd'].labels_
            for scores in self.scores:
                s = scores(x_pca, y)
                #print(f'{scores.__name__}: {s}')
            self.combine3['hdbscan labels'] = y
            self.combine3['hdbscan identifier'] = (self.combine3['hdbscan labels'] == -1).astype(int)
            self.combine3['hdbscan category'] = self.combine3['hdbscan identifier'].apply(lambda x: 'Anomaly' if x==1 else 'Not anomaly')

            self.hd_iso_pipeline.fit(self.copy)
            x_iso = self.hd_iso_pipeline.named_steps['preprocessor'].transform(self.copy)
            x_iso_pca = self.hd_iso_pipeline.named_steps['pca'].transform(x_iso)

            x_iso_fit = self.hd_iso_pipeline.named_steps['iso'].fit(x_iso_pca)
            x_iso_predict = self.hd_iso_pipeline.named_steps['iso'].predict(x_iso_pca)
            x_iso_scores = self.hd_iso_pipeline.named_steps['iso'].decision_function(x_iso_pca)
            hd_iso_scores_df = pd.DataFrame({'hdbscan isolation scores': x_iso_scores,
                                             'hdbscan isolation detector': x_iso_predict})
            hd_iso_scores_df.insert(0, 'id', hd_iso_scores_df.index+1)
            hd_iso_scores_df.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_hdbscan_iso.csv', index=False)

            self.combine3['hdbscan isolation labels'] = x_iso_predict
            self.combine3['hdbscan isolation anomaly identifier'] = (self.combine3['hdbscan isolation labels'] == -1).astype(int)
            self.combine3['hdbscan isolation anomaly category'] = self.combine3['hdbscan isolation anomaly identifier'].apply(lambda x: 'Anomaly' if x==1 else 'Not anomaly')
            self.combine3.insert(0, 'id', self.combine3.index+1)
            self.combine3.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file3.csv', index=False)

            self.total_identifier = pd.Series(
                        self.combine3['dbscan isolation anomaly identifier'].astype(int) +
                        self.combine3['optics isolation anomaly identifier'].astype(int) +
                        self.combine3['hdbscan isolation anomaly identifier'].astype(int))

            self.total_category = self.total_identifier.map({0: 'none',1: 'low', 2: 'medium', 3: 'critical'})

            self.combine3['anomaly identifier'] = self.total_identifier
            self.combine3['anomaly detector'] = self.total_category
            self.combine3.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file3.csv', index=False)

            print(self.combine3.head().to_string())
        except Exception as e: print(f'invalid hdbscan: {e}')

    def flag_fraud(self):
        try:
            self.final_fraud = pd.read_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file3.csv')
            self.final_fraud['fraud identifier'] = self.final_fraud['anomaly detector'].apply(lambda x: 1 if x == 'critical' else 0)
            self.final_fraud['fraud detector'] = self.final_fraud['fraud identifier'].apply(lambda x: 'is fraud' if x == 1 else 'not fraud')
            self.final_fraud.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file4_before_drop_isna.csv', index=False)

            self.final_drop_fraud = self.final_fraud.drop(index=[44,45,46]).reset_index(drop=True)

            id_count = self.final_drop_fraud['id'].count()
            f_identifier = self.final_drop_fraud['fraud identifier'].sum()
            self.final_drop_fraud['fraud measures'] = f_identifier / id_count
            print(self.final_drop_fraud.head().to_string())
            self.final_drop_fraud.to_csv('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file4.csv', index=False)
            #print(self.final_drop_fraud.dtypes)
            #print(self.final_drop_fraud.head().to_string())





        except Exception as e: print(f'invalid flag column: {e}')



if __name__ == "__main__":
    model_name = input('Enter model name here: ')
    cf = ClusterF('c:/Users/anton/OneDrive/gov_finance/gov_clean_cluster.csv', model_name)
    if model_name == 'd':
        cf.db_scan()
    elif model_name == 'o':
        cf.op_tics()
    elif model_name == 'h':
        cf.hd_scan()
    elif model_name == 'f':
        cf.flag_fraud()
