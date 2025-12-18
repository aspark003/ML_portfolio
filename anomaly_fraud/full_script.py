import os
from zipfile import BadZipFile, ZipFile
import openpyxl
import pandas as pd

class FileLoader:
    def __init__(self, file=None):
        self.file = file
        self.df = None

    def load(self, file=None):
        # Update file path if a new one is provided
        if file:
            self.file = file

        if not self.file:
            raise ValueError("File path not available")

        # Get file extension
        extension = os.path.splitext(self.file)[1].lower()

        try:
            # ---------- Excel ----------
            if extension == ".xlsx":
                try:
                    # Ensure the file is a proper zip (xlsx)
                    with ZipFile(self.file, 'r') as zip_ref:
                        corrupt_file = zip_ref.testzip()
                        if corrupt_file:
                            raise BadZipFile(f"Corrupt file inside archive: {corrupt_file}")
                    # File is valid Excel, load it
                    self.df = pd.read_excel(self.file, engine='openpyxl')
                    print("Excel file loaded successfully.")

                except BadZipFile:
                    raise ValueError(f"Excel file is corrupted or not a valid .xlsx: {self.file}")

            # ---------- CSV ----------
            elif extension == ".csv":
                self.df = pd.read_csv(self.file)
                print("CSV file loaded successfully.")

            # ---------- TSV ----------
            elif extension == ".tsv":
                self.df = pd.read_csv(self.file, sep="\t")
                print("TSV file loaded successfully.")

            # ---------- JSON ----------
            elif extension in [".json", ".jsonl"]:
                try:
                    if extension == ".jsonl":
                        self.df = pd.read_json(self.file, lines=True)
                    else:
                        self.df = pd.read_json(self.file)
                    print("JSON file loaded successfully.")
                except ValueError as e:
                    raise ValueError(f"Invalid JSON file: {e}")

            # ---------- TXT ----------
            elif extension in [".txt", ".text"]:
                with open(self.file, "r", encoding="utf-8", errors="ignore") as f:
                    self.df = pd.DataFrame([line.strip() for line in f.readlines()], columns=["text"])
                print("Text file loaded successfully.")

            else:
                raise ValueError(f"Unsupported file type: {extension}")

        except Exception as e:
            raise ValueError(f"Failed to load file: {e}")

        # Print first few rows
        if self.df is not None:
            print(self.df.head().to_string())

        #self.df.to_csv('c:/Users/anton/OneDrive/fraud_detection/gl1.csv', index=False)
        return self.df


from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest

class DModel:
    def __init__(self, file, header=3):
        self.df = pd.read_csv(file, encoding='utf-8-sig', engine='python', header=header)
        self.df = self.df.drop(self.df.index[58:]).reset_index(drop=True)
        self.df.columns = self.df.columns.str.lower().str.strip()
        self.df.insert(0, 'id', self.df.index+1)
        self.df.to_csv('c:/Users/anton/OneDrive/fraud_detection/original.csv', index=False)

        self.copy = self.df.copy()

        self.num = self.copy.select_dtypes(include=['number', 'complex']).columns.tolist()
        self.obj = self.copy.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)

        self.n_impute = Pipeline([('num', SimpleImputer(strategy='median')),
                                  ('number', self.scaler)])
        self.o_impute = Pipeline([('obj', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('object', self.encoder)])

        self.preprocessor = ColumnTransformer([('scaler', self.n_impute, self.num),
                                               ('encoder', self.o_impute, self.obj)])

        self.d_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=2)),
                                    ('dbscan', DBSCAN(eps=0.2, min_samples=5))])

        self.o_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=8)),
                                    ('optics', OPTICS(min_samples=4, xi=0.05))])

        self.h_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=2)),
                                    ('hdbscan', HDBSCAN(min_samples=5, min_cluster_size=5))])

        self.i_pipeline = Pipeline([('preprocessor', self.preprocessor),
                                    ('pca', PCA(n_components=2)),
                                    ('iso', IsolationForest(contamination=0.2, random_state=42))])

        self.scores = ([silhouette_score, calinski_harabasz_score, davies_bouldin_score])

        self.d_pipeline.fit(self.copy)
        x = self.d_pipeline.named_steps['preprocessor'].transform(self.copy)
        x_pca = self.d_pipeline.named_steps['pca'].transform(x)
        variance = self.d_pipeline.named_steps['pca'].explained_variance_ratio_
        c_sum = np.cumsum(variance)

        y = self.d_pipeline.named_steps['dbscan'].labels_

        self.df['dbscan labels'] = y
        self.df['dbscan identifier'] = (self.df['dbscan labels'] == -1).astype(int)
        self.df['dbscan category'] = self.df['dbscan identifier'].apply(lambda x: 'anomaly' if x == 1 else 'not anomaly')
        #self.df.to_csv('c:/Users/anton/OneDrive/fraud_detection/original.csv', index=False)

        #print(self.df.head().to_string())

        for score in self.scores:
            s = score(x_pca, y)
            print(f'dbscan: {score.__name__}:{s}')
        print()

        self.o_pipeline.fit(self.copy)
        xo = self.o_pipeline.named_steps['preprocessor'].transform(self.copy)
        xo_pca = self.o_pipeline.named_steps['pca'].transform(xo)
        yl = self.o_pipeline.named_steps['optics'].labels_
        for o_core in self.scores:
            o = o_core(xo_pca, yl)
            print(f'optics: {o_core.__name__}:{o}')
        print()

        self.df['optics labels'] = yl
        self.df['optics identifier'] = (self.df['optics labels'] == -1).astype(int)
        self.df['optics category'] = self.df['optics identifier'].apply(lambda x: 'anomaly' if x == 1 else 'not anomaly')
        #self.df.to_csv('c:/Users/anton/OneDrive/fraud_detection/original.csv', index=False)


        self.h_pipeline.fit(self.copy)
        xh = self.h_pipeline.named_steps['preprocessor'].transform(self.copy)
        xh_pca = self.h_pipeline.named_steps['pca'].transform(xh)
        yh = self.h_pipeline.named_steps['hdbscan'].labels_

        for h_score in self.scores:
            h = h_score(xh_pca, yh)
            print(f'hdbscan: {h_score.__name__}:{h}')
        print()

        self.df['hdbscan labels'] = yh
        self.df['hdbscan identifier'] = (self.df['hdbscan labels'] == -1).astype(int)
        self.df['hdbscan category'] = self.df['hdbscan identifier'].apply(lambda x: 'anomaly' if x == 1 else 'not anomaly')

        self.df['anomaly agreement strength'] = ((self.df['dbscan identifier']).astype(int)+
                                        (self.df['optics identifier']).astype(int)+
                                        (self.df['hdbscan identifier'].astype(int)))

        self.df['anomaly category'] = self.df['anomaly agreement strength'].map({3: 'high',2: 'medium', 1: 'low', 0: 'none'})


        self.df['investigation required'] = self.df['anomaly agreement strength'].apply(lambda x: 'needs review' if x >= 2 else 'none')
        self.df['fraud candidate'] = self.df['investigation required'].apply(lambda x: 'potential fraud' if x == 'needs review' else 'none')


        self.df.to_csv('c:/Users/anton/OneDrive/fraud_detection/original.csv', index=False)
        print(self.df.head().to_string())

        #print(self.copy.head().to_string())




if __name__ == "__main__":
    model_name = input('enter model name here: ')
    fc = FileLoader("c:/Users/anton/OneDrive/fraud_detection/sof_gl_test.xlsx")
    if model_name == 'a':
        fc.load()

    dm = DModel('c:/Users/anton/OneDrive/fraud_detection/gl1.csv')











