import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from yellowbrick.regressor import PredictionError, ResidualsPlot, AlphaSelection, CooksDistance


class ClusterRegression:
    def __init__(self, file_path, model_name):
        self.model_name = model_name
        self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python')
        self.drop_df = self.df.copy()
        self.copy = self.df.copy()
        self.copy = self.copy.drop(columns=['id', 'project', 'gl organization', 'ledger name', 'dbscan category', 'dbscan isolation anomaly category', 'optics category', 'optics isolation anomaly category', 'hdbscan category', 'hdbscan isolation anomaly category', 'anomaly detector','fraud detector'])

        self.scaler = StandardScaler()
        self.num = self.copy.select_dtypes(include=['number', np.number]).columns.tolist()

        self.scores = ([r2_score, mean_absolute_error, mean_squared_error])



    def linear_model(self):
        try:
            X = self.copy.drop(columns=['obligations'])
            y= self.copy['obligations']

            lr = LinearRegression()
            lr.fit(X,y)
            lr_predict_x = lr.predict(X)
            #print(lr_predict_x)

            l_cvs = cross_val_score(lr, X,y, cv=5, scoring='r2').mean()
            #print(f'mean cross val scores: {l_cvs}')
            for l_s in self.scores:
                l = l_s(y, lr_predict_x)
                #print(f'{l_s.__name__}:{l}')


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


            lr.fit(X_train, y_train)
            x_predict = lr.predict(X_test)
            lr_dict = pd.DataFrame({'actual': y_test,
                                    'predict': x_predict})
            lr_dict = lr_dict.reset_index(drop=True)
            #print(f'{lr_dict}')
            for scores in self.scores:
                s = scores(y_test, x_predict)
                #print(f'{scores.__name__}: {s:.10f}')

            cvs = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()
            #print(f'mean cross val score: {cvs}')

            #PredictionError, ResidualsPlot, AlphaSelection, CooksDistance

            viz = PredictionError(lr)
            v = viz.fit(X_train, y_train)
            #v.show()

            riz = ResidualsPlot(lr)
            r = riz.fit(X_train, y_train)
            #r.show()

            x_const = sm.add_constant(X)

            x_ols = sm.OLS(y, x_const)
            x_fit = x_ols.fit()

            model_influence = x_fit.get_influence()

            cooks_d, p_value = model_influence.cooks_distance

            n = x_const.shape[0]
            p = x_const.shape[1]
            threshold = 4 / (n-p)
            influential_idx = np.where(cooks_d > threshold)[0]
            cooks_series = pd.Series(cooks_d)

            cooks_max = cooks_d[np.isfinite(cooks_d)].max()
            update_cooks = cooks_series.replace([np.inf, -np.inf, np.nan], cooks_max)
            cooks_series = pd.Series(update_cooks)

            cooks_identifier = cooks_series.index.isin(influential_idx).astype(int)
            self.drop_df['LINEAR cooks distance score'] = update_cooks
            self.drop_df['LINEAR cooks distance identifier'] = cooks_identifier
            #print(self.drop_df.head().to_string())
            self.drop_df['LINEAR cooks category'] = (self.drop_df['LINEAR cooks distance identifier'].apply(lambda x: 'Outlier' if x == 1 else 'Not outlier'))

            self.drop_df.to_csv('c:/Users/anton/OneDrive/gov_regression1.csv', index=False)
            print(self.drop_df.head().to_string())
        except Exception as e:
            print(f'invalid linear model: {e}')

    def ridge_model(self):
        try:

            self.drop_df1 = pd.read_csv('c:/Users/anton/OneDrive/gov_regression1.csv')
            #print(self.drop_df1)
            X = self.copy.drop(columns=['obligations'])
            y = self.copy['obligations']

            rg = Ridge(alpha=1.0)
            rg.fit(X, y)
            rg_predict_x = rg.predict(X)
            # print(rg_predict_x)

            r_cvs = cross_val_score(rg, X, y, cv=5, scoring='r2').mean()

            cvs = cross_val_score(rg, X, y, cv=5, scoring='r2').mean()
            # print(f'mean cross val score: {cvs}')


            x_const = sm.add_constant(X)

            x_ols = sm.OLS(y, x_const)
            x_fit = x_ols.fit()

            model_influence = x_fit.get_influence()

            cooks_d, p_value = model_influence.cooks_distance

            n = x_const.shape[0]
            p = x_const.shape[1]
            threshold = 4 / (n - p)
            influential_idx = np.where(cooks_d > threshold)[0]
            cooks_series = pd.Series(cooks_d)

            cooks_max = cooks_d[np.isfinite(cooks_d)].max()
            update_cooks = cooks_series.replace([np.inf, -np.inf, np.nan], cooks_max)
            cooks_series = pd.Series(update_cooks)
            #print(self.copy.head().to_string())
            cooks_identifier = cooks_series.index.isin(influential_idx).astype(int)
            #print(self.drop_df1.head().to_string())
            self.drop_df1['RIDGE cooks distance score'] = update_cooks
            self.drop_df1['RIDGE cooks distance identifier'] = cooks_identifier
            # print(self.drop_df.head().to_string())

            self.drop_df1['RIDGE cooks distance category'] = self.drop_df1['RIDGE cooks distance identifier'].apply(lambda x: 'Outlier' if x==1 else 'Not outlier')

            print(self.drop_df1.head().to_string())

            self.drop_df1.to_csv('c:/Users/anton/OneDrive/gov_regression2.csv', index=False)

        except Exception as e: print(f'invalid ridge: {e}')


    def lasso_model(self):
        try:
            self.drop_df2 = pd.read_csv('c:/Users/anton/OneDrive/gov_regression2.csv')
            X =  self.copy.drop(columns = ['obligations'])
            y = self.copy['obligations']
            la = Lasso(alpha=1.0)
            la.fit(X, y)
            rg_predict_x = la.predict(X)
            # print(rg_predict_x)

            r_cvs = cross_val_score(la, X, y, cv=5, scoring='r2').mean()

            cvs = cross_val_score(la, X, y, cv=5, scoring='r2').mean()
            # print(f'mean cross val score: {cvs}')


            x_const = sm.add_constant(X)

            x_ols = sm.OLS(y, x_const)
            x_fit = x_ols.fit()

            model_influence = x_fit.get_influence()

            cooks_d, p_value = model_influence.cooks_distance

            n = x_const.shape[0]
            p = x_const.shape[1]
            threshold = 4 / (n - p)
            influential_idx = np.where(cooks_d > threshold)[0]
            cooks_series = pd.Series(cooks_d)

            cooks_max = cooks_d[np.isfinite(cooks_d)].max()
            update_cooks = cooks_series.replace([np.inf, -np.inf, np.nan], cooks_max)
            cooks_series = pd.Series(update_cooks)

            cooks_identifier = cooks_series.index.isin(influential_idx).astype(int)
            self.drop_df2['LASSO cooks distance score'] = update_cooks
            self.drop_df2['LASSO cooks distance identifier'] = cooks_identifier
            # print(self.drop_df.head().to_string())
            self.drop_df2['LASSO cooks distance category'] = (
                self.drop_df2['LASSO cooks distance identifier'].apply(lambda x: 'Outlier' if x == 1 else 'Not outlier'))
            print(self.drop_df2.head().to_string())

            self.drop_df2.to_csv('c:/Users/anton/OneDrive/gov_regression3.csv', index=False)

        except Exception as e: print(f'invalid ridge: {e}')

    def reg_final(self):
        try:
            self.drop_df3 = pd.read_csv('c:/Users/anton/OneDrive/gov_regression3.csv')

            self.drop_df3['Regression final identifier'] = ((self.drop_df3['fraud identifier'].astype(int)+
                                                             (self.drop_df3['LINEAR cooks distance identifier'].astype(int)+
                                                              (self.drop_df3['RIDGE cooks distance identifier'].astype(int))+
                                                              (self.drop_df3['LASSO cooks distance identifier'].astype(int)))))
            self.drop_df3['Regression final category'] = self.drop_df3['Regression final identifier'].map({4: 'critical', 3: 'high', 2: 'medium', 1: 'low', 0: 'none'})
            id_count = self.drop_df3['id'].count()

            f_identifier = self.drop_df3['Regression final identifier'].sum()
            self.drop_df3['Regression final measures'] = f_identifier / id_count


            self.drop_df3.to_csv('c:/Users/anton/OneDrive/gov_regression4.csv', index=False)
            print(self.drop_df3.head().to_string())
        except Exception as e: print(f'invalid final: {e}')


if __name__ == "__main__":
    model_name = input('Enter model name here: ')
    cr = ClusterRegression('c:/Users/anton/OneDrive/gov_finance/gov_soft_gl_auto_dash_file4.csv', model_name)
    if model_name == 'l':
        cr.linear_model()
    elif model_name == 'r':
        cr.ridge_model()
    elif model_name == 'la':
        cr.lasso_model()
    elif model_name == 'f':
        cr.reg_final()
