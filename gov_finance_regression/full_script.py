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

        self.df.to_csv('c:/Users/anton/OneDrive/practice/practice1.csv', index=False)
        return self.df

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class CleanFile:
    def __init__(self, file1, header=2):
        self.file1 = file1
        self.file1 = pd.read_csv(file1, encoding='utf-8-sig', engine='python', header=header)
        self.file1.columns = self.file1.columns.str.replace('-', ' ', regex=True).str.lower().str.strip()
        self.file1 = self.file1.drop(self.file1.index[474:])
        self.file1.insert(0, 'id', self.file1.index+1)
        self.file1.to_csv('c:/Users/anton/OneDrive/practice/practice_original.csv', index=False)

    def clean(self):
        try:
            self.num = self.file1.select_dtypes(include=['number', 'complex']).columns.tolist()
            self.obj = self.file1.select_dtypes(include=['object', 'string', 'bool', 'category']).columns.tolist()

            obj_simple = SimpleImputer(strategy='constant', fill_value='missing')
            num_simple = SimpleImputer(strategy='median')


            self.preprocessor = ColumnTransformer([('obj', obj_simple, self.obj),
                                                   ('num', num_simple, self.num)])

            self.preprocessor.fit(self.file1)
            self.file2 = self.preprocessor.transform(self.file1)
            self.file2 = pd.DataFrame(self.file2, columns=self.preprocessor.get_feature_names_out())
            self.file2.insert(0, 'id', self.file2.index+1)

            self.file2.columns = self.file2.columns.str.replace('obj__', '', regex=True).str.lower().str.strip()
            self.file2.columns = self.file2.columns.str.replace('num__', '', regex=True).str.lower().str.strip()

            print(self.file2.dtypes)
            self.file2.to_csv('c:/Users/anton/OneDrive/practice/practice0.csv', index=False)
            print(self.file2.head().to_string())
        except Exception as e: print(f'file not cleaned: {e}')

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

class LinearModel:
    def __init__(self, file):
        self.file3 = pd.read_csv(file,encoding='utf-8-sig', engine='python')
        self.scores = ([r2_score, mean_absolute_error, mean_squared_error])

    def linear_reg(self):
        try:
            self.file3 = self.file3.drop(columns=['id', 'task organization', 'bfy', 'ba bsa bli', 'fund', 'project number', 'task number', 'expenditure type', 'limit', 'budget authority', 'commitments']).reset_index(drop=True)
            X = self.file3.drop(columns=['funds used'])
            y = self.file3['funds used']

            mm = MinMaxScaler()
            X = mm.fit_transform(X)
            lr = LinearRegression()

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
            lr.fit(X_train, y_train)
            lr_test = lr.predict(X_test)
            for sc in self.scores:
                c = sc(y_test, lr_test)
                print(f'train / test scores: {sc.__name__}: {c}')
            #print()

            # no training, straight to X, y
            lr.fit(X, y)
            lr_predict = lr.predict(X)
            for scores in self.scores:
                s = scores(y, lr_predict)
                #print(f'{scores.__name__}: {s}')

            cvs = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()
            #print(f'mean cross val score: {cvs}')
            #print()

            las = LassoCV()
            las.fit(X_train, y_train)
            las_predict = las.predict(X_test)

            for l in self.scores:
                lu = l(y_test, las_predict)
                #print(f'lasso train score: {l.__name__}:{lu}')
            #print()

            lvs = cross_val_score(las, X, y, cv=5, scoring='r2').mean()
            #print(f'mean lasso cross val: {lvs}')
            #print()

            las.fit(X, y)
            lasp = las.predict(X)
            for lu in self.scores:
                luu = lu(y,lasp)
                #print(f'Lasso normal x,y scores: {luu}')

            rid = RidgeCV()
            rid.fit(X_train, y_train)
            r_test = rid.predict(X_test)
            for rsc in self.scores:
                r = rsc(y_test, r_test)
                #print(f'ridge: {rsc.__name__}: {r}')
            #print()

            rcvs = cross_val_score(rid, X, y, cv=5, scoring='r2').mean()
            #print(f'mean ridge cross val scores: {rcvs}')
            #print()

            rid.fit(X,y)
            d = rid.predict(X)
            for rd in self.scores:
                rids = rd(y,d)
                #print(f'ridge: {rd.__name__}:{rids}')
            self.o = pd.read_csv('c:/Users/anton/OneDrive/practice/practice_original.csv')

            self.o['linear actual'] = y
            self.o['linear predict'] = d

            self.o['linear final score'] =self.o['linear actual'] - self.o['linear predict']

            high = (self.o['linear final score'].quantile(0.75))
            low = (self.o['linear final score']).quantile(0.25)

            self.o['linear residual'] = self.o['linear final score'].apply(lambda x: 3 if x > high else 1 if x <= low else 2)
            self.o['linear residual category'] = self.o['linear residual'].map({3: 'over',2: 'normal', 1: 'under'})

            self.o['residual severity'] = self.o['linear residual'].apply(lambda x: 'review required' if x > 2 else 'none')
            self.o['investigation required'] = self.o['residual severity'].apply(lambda x: 'investigate' if x == 'review required' else 'none')

            self.o.to_csv('c:/Users/anton/OneDrive/practice/practice_original.csv', index=False)
            print(self.o.head().to_string())
        except Exception as e:print(f'invalid linear model: {e}')

if __name__ == "__main__":
    model_name = input('enter model name here: ')
    fc = FileLoader("c:/Users/anton/OneDrive/practice/sof_pt_auto_practice.xlsx")
    if model_name == 'a':
        df = fc.load()

    cf = CleanFile('c:/Users/anton/OneDrive/practice/practice1.csv')
    if model_name == 'b':
        cf.clean()

    lm = LinearModel('c:/Users/anton/OneDrive/practice/practice0.csv')
    if model_name == 'c':
        lm.linear_reg()



