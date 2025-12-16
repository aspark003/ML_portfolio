import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

class GovRegression:
    def __init__(self, file_path, model_name):
        try:

            self.start1 = pd.read_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto3.csv')
            #print(self.start1.head().to_string())
            self.model_name = model_name
            self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python')
            self.df.columns = self.df.columns.str.strip()
            self.copy = self.df.copy()

            self.num = self.copy.select_dtypes(include=['number', 'complex']).columns.tolist()
            self.obj = self.copy.select_dtypes(include=['object', 'string', 'bool', 'category']).columns.tolist()

            self.scaler = MinMaxScaler()
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

            self.num_input = Pipeline([('num_input', SimpleImputer(strategy='median')),
                                       ('scaler', self.scaler)])

            self.obj_input = Pipeline([('obj_input', SimpleImputer(strategy='constant', fill_value='missing')),
                                       ('encoder', self.encoder)])

            self.preprocessor = ColumnTransformer([('scaler', self.num_input, self.num),
                                                   ('encoder', self.obj_input, self.obj)])

            self.scores = ([r2_score, mean_absolute_error, mean_squared_error])


        except Exception as e: print(f'invalid file: {e}')

    def linear_reg(self):
        try:
            self.copy = self.preprocessor.fit_transform(self.copy)
            self.copy1 = pd.DataFrame(self.copy, columns=self.preprocessor.get_feature_names_out())

            self.copy1.columns = self.copy1.columns.str.replace('scaler__', '', regex=True).str.lower().str.strip()
            self.X = self.copy1.drop(columns=['funds used'])
            self.y = self.copy1['funds used']
            lr = LinearRegression()

            X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=42)

            lr.fit(X_train, y_train)
            lr_predict = lr.predict(X_test)
            l_df = pd.DataFrame({'actual': y_test,
                                 'predict': lr_predict}).reset_index(drop=True)
            #print(l_df.head())
            for score in self.scores:
                s = score(y_test, lr_predict)
                print(f'{score.__name__}: {s}')

            cvs = cross_val_score(lr, self.X,self.y, cv=5, scoring='r2').mean()

            print(f'mean cross val score: {cvs}')
        except Exception as e:
            print(f'invalid linear regression model: {e}')

    def random_forest(self):
        try:
            #print(self.start1.head().to_string())
            self.start2 = pd.read_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto6.csv')

            #print(self.start2.shape)
            #print(self.start2.head().to_string())
            self.copy = self.preprocessor.fit_transform(self.copy)
            self.copy1 = pd.DataFrame(self.copy, columns=self.preprocessor.get_feature_names_out())

            self.copy1.columns = self.copy1.columns.str.replace('scaler__', '', regex=True).str.lower().str.strip()
            X = self.copy1.drop(columns=['funds used'])
            y = self.copy1['funds used']


            rf = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=3,random_state=42)

            rf.fit(X,y)
            rf_predict = rf.predict(X)
            self.start2['actual funds used'] = y

            self.start1['actual funds used'] = y
            self.start1['predicted funds used'] = rf_predict
            self.start2['predicted funds used'] = rf_predict
            residual = (self.start1['actual funds used'] - self.start1['predicted funds used'])
            residual = (self.start2['actual funds used'] - self.start2['predicted funds used'])
            self.start1['residual'] = residual
            self.start2['residual'] = residual
            #print(self.start1.head().to_string())
            self.start1.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto4.csv',index=False)

            rf_df = pd.DataFrame({'actual': y,
                                  'predict': rf_predict}).reset_index(drop=True)

            for score in self.scores:
                s = score(y, rf_predict)
                #print(f'{score.__name__}: {s}')

            r2 = r2_score(y, rf_predict)
            mae = mean_absolute_error(y,rf_predict)
            mse = mean_squared_error(y,rf_predict)
            cvs = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()
            #print(f'r2: {r2}, mae: {mae}, mse: {mse}, mean cross val scores: {cvs}')

            # example thresholds
            print(self.start1['residual'].describe())

            q1 = self.start1['residual'].quantile(0.25)
            q3 = self.start1['residual'].quantile(0.75)

            self.start1['residual label'] = np.select(
                [self.start1['residual'] > q3,self.start1['residual'] >= q1],[3, 2],default=1)
            self.start1['residual category'] = self.start1['residual label'].apply(lambda x: 'high' if x == 3 else 'medium' if x == 2 else 'low' if x == 1 else 0)
            print(self.start1.head().to_string())


            self.start1.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto5.csv', index=False)

            q1 = self.start2['residual'].quantile(0.25)
            q3 = self.start2['residual'].quantile(0.75)

            self.start2['residual label'] = np.select(
                [self.start2['residual'] > q3, self.start2['residual'] >= q1], [3, 2], default=1)
            self.start2['residual category'] = self.start2['residual label'].apply(
                lambda x: 'high' if x == 3 else 'medium' if x == 2 else 'low' if x == 1 else 0)

            #print(self.start1.head().to_string())
            self.start2 = self.start2.drop(index=[474]).reset_index(drop=True)

            self.start2.to_csv('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto6.csv', index=False)

        except Exception as e: print(f'invalid random forest model: {e}')



if __name__ == "__main__":
    model_name = input('Enter model name here: ')
    gr = GovRegression('c:/Users/anton/OneDrive/gov_finance_regression_model/gov_pt_auto2.csv', model_name)
    if model_name == 'l':
        gr.linear_reg()
    elif model_name == 'r':
        gr.random_forest()
