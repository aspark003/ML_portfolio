import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

class CleanFile:
    def __init__(self, file,header=2):
        try:
            self.df = pd.read_csv(file, encoding='utf-8-sig', engine='python', header=header)
            self.df.columns = self.df.columns.str.replace('-', ' ', regex=True).str.lower().str.strip()
            self.df.insert(0, 'id', self.df.index+1)
            self.df = self.df.drop(self.df.index[473:]).reset_index(drop=True)

            self.df1 = self.df.copy()

            self.use = self.df1.copy()

            self.use = self.use.drop(columns=['task organization', 'bfy', 'ba bsa bli', 'fund', 'limit', 'project number', 'task number', 'expenditure type', 'class category', 'class code', 'budget authority', 'commitments', 'obligations', 'non labor expenditures'])

            self.use.to_csv('c:/Users/anton/OneDrive/test1.csv', index=False)
            self.use1 = self.use.copy()
            self.scores = ([r2_score, mean_absolute_error, mean_squared_error])
        except Exception as e: print(f'invalid file: {e}')

    def linear_model(self):
        try:
            si = SimpleImputer(strategy='median')
            self.use2 = si.fit_transform(self.use1)
            self.use2 = pd.DataFrame(self.use2, columns=self.use1.columns)
            #self.use2.to_csv('c:/Users/anton/OneDrive/test3.csv', index=False)

            X = self.use2.drop(columns='funds used')
            y = self.use2['funds used']

            mm = MinMaxScaler()


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train = mm.fit_transform(X_train)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            X_test = mm.transform(X_test)
            lr_predict = lr.predict(X_test)

            for scores in self.scores:
                s = scores(y_test, lr_predict)
                #print(f'{scores.__name__}:{s}')

            cvs = cross_val_score(lr, X,y, cv=8, scoring='r2').mean()
            #print(f'mean cross val score: {cvs}')

            la = LassoCV()
            la.fit(X_train, y_train)
            lasso_predict = la.predict(X_test)
            for sc in self.scores:
                sa = sc(y_test, lasso_predict)
                #print(f'{sc.__name__}:{sa}')

            ri = RidgeCV()
            ri.fit(X_train,y_train)
            ri_predict = ri.predict(X_test)
            for rs in self.scores:
                r =rs(y_test, ri_predict)
                #print(f'{rs.__name__}:{r}')

            cri = cross_val_score(ri, X,y, cv=5, scoring='r2').mean()
            #print(f'mean cross val score: {cri}')

            self.df1['actual linear score'] = y_test
            pre = pd.Series(ri_predict)
            self.df1['predicted linear score'] = pre

            act_pre = (self.df1['actual linear score'] - self.df1['predicted linear score'])
            self.df1['residual'] = act_pre


            top = self.df1['residual'].quantile(0.75)

            bottom = self.df1['residual'].quantile(0.25)
            self.df1['residual labels'] = np.select(
                [
                    self.df1['residual'] >= top,
                    (self.df1['residual'] >= bottom) & (self.df1['residual'] < top)],[3,2],default=1)

            self.df1['residual category'] = self.df1['residual labels'].map({3: 'high', 2:'average', 1: 'low'})


            #print(self.df1['residual'].describe())

            #print(self.df1.head().to_string())
            self.df1.to_csv('c:/Users/anton/OneDrive/test0.csv', index=False)

            random_f = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=4, random_state=42)
            random_f.fit(X_train, y_train)
            ra_pre = random_f.predict(X_test)
            for ra_score in self.scores:
                ras = ra_score(y_test, ra_pre)
                #print(f'{ra_score.__name__}:{ras}')

            c_v = cross_val_score(random_f,X,y, cv=5, scoring='r2').mean()
            #print(f'mean cross val score: {c_v}')

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            gs = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='r2')
            gs.fit(X_train, y_train)
            gs_predict = gs.predict(X_test)
            for gcore in self.scores:
                gc = gcore(y_test, gs_predict)
                #print(f'{gcore.__name__}:{gc}')

            gsv = cross_val_score(gs, X, y, cv=5, scoring='r2').mean()
            print(f'mean cross val scores: {gsv}')

            self.df1['gridsearch actual score'] = y_test
            g_pre = pd.Series(gs_predict)
            self.df1['gridsearch predict score'] = g_pre

            act_pre = (self.df1['gridsearch actual score'] - self.df1['gridsearch predict score'])
            self.df1['gridsearch residual'] = act_pre

            top = self.df1['gridsearch residual'].quantile(0.75)

            bottom = self.df1['gridsearch residual'].quantile(0.25)

            self.df1['gridsearch labels'] = np.select([self.df1['gridsearch residual'] >=top, (self.df1['gridsearch residual']<top)], [3,2], default=1)
            self.df1['gridsearch category'] = self.df1['gridsearch labels'].map({3: 'high', 2: 'average', 1: 'low'})

            self.df1.to_csv('c:/Users/anton/OneDrive/test0.csv', index=False)

            xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            xgb.fit(X_train,y_train)
            xe_predict = xgb.predict(X_test)

            for xscore in self.scores:
                xu = xscore(y_test, xe_predict)
                print(f'{xscore.__name__}:{xu}')

            xcvs = cross_val_score(xgb, X, y, cv=5, scoring='r2').mean()
            print(f'mean cross val score: {xcvs}')

            self.df1['xgb actual score'] = y_test
            xpre = pd.Series(xe_predict)
            self.df1['xgb predict score'] = xpre

            a = self.df1['xgb actual score']
            b = self.df1['xgb predict score']
            self.df1['xgb final score'] = a - b

            top = self.df1['xgb final score'].quantile(0.75)
            low = self.df1['xgb final score'].quantile(0.25)

            self.df1['xgb labels'] = np.select([self.df1['xgb final score'] >= top, (self.df1['xgb final score'] < top)], [3, 2], default=1)

            self.df1['xgb category'] = self.df1['xgb labels'].map({3: 'high', 2: 'average', 1: 'low'})

            print(self.df1.head().to_string())

            t_top = self.df1['xgb final score'].quantile(0.75)
            b_low = self.df1['xgb final score'].quantile(0.25)
            self.df1['final label'] = np.select([self.df1['xgb final score']>=t_top, (self.df1['xgb final score']<t_top)], [3, 2], default=1)
            self.df1['final category'] = self.df1['final label'].map({3: 'high', 2: 'average', 1: 'low'})

            self.df1.to_csv('c:/Users/anton/OneDrive/test0.csv', index=False)
        except Exception as e: print(f'invalid linear model: {e}')

if __name__ == "__main__":

    cf = CleanFile('c:/Users/anton/OneDrive/test11.csv')
    cf.linear_model()
