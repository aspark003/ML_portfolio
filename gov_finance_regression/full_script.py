import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor


class B:
    def __init__(self, file):
        self.df = pd.read_csv(file, encoding='utf-8-sig', engine='python')
        self.df.columns = self.df.columns.str.replace('_', ' ', regex=True).str.lower().str.strip()
        self.copy1 = self.df.copy()
        #self.df.to_csv('c:/Users/anton/supervised/final1.csv', index=False)
        self.df1 = pd.read_csv('c:/Users/anton/supervised/final1.csv')


        x = self.df.drop(columns=['annual income', 'region', 'account type'])
        y = self.df['annual income']

        mm = MinMaxScaler()
        feature = mm.fit_transform(x)

        lr = LinearRegression()
        lr.fit(feature, y)

        # IMPORTANT — predict on the SAME scaled data
        lr_pre = lr.predict(feature)

        y_test_reset = y.reset_index(drop=True)

        lr_dict = pd.DataFrame({'linear actual': y_test_reset,
                                'linear predict': lr_pre})

        lr_dict['linear residual'] = lr_dict['linear actual'] - lr_dict['linear predict']

        lr_dict['residual'] = np.where(
            lr_dict['linear actual'] > lr_dict['linear predict'],
            'under',
            'over'
        )
        print(lr_dict.head().to_string())
        lin_r2 = r2_score(y, lr_pre)
        lin_mae = mean_absolute_error(y, lr_pre)
        lin_mse = mean_squared_error(y, lr_pre)
        lin_cvs = cross_val_score(lr, feature, y, cv=5, scoring='r2').mean()

        print('linear regression r2 score: ', lin_r2)
        print('linear regression mean absolute error: ', lin_mae)
        print('linear regression mean squared error: ', lin_mse)
        print()
        print('linear regression cross val score: ', lin_cvs)
        print()

        self.df1 = self.df1.join(lr_dict)

        x1 = self.copy1.drop(columns=['annual income', 'region', 'account type'])
        y1 = self.copy1['annual income']

        mm = MinMaxScaler()
        feature = mm.fit_transform(x)

        en = RandomForestRegressor(random_state=42)
        en.fit(feature, y1)

        rg_pre = en.predict(feature)

        y_test_reset = y1.reset_index(drop=True)

        lr_dict = pd.DataFrame({'random forest actual': y_test_reset,
                                'random forest prediction': rg_pre})

        lr_dict['random forest residual'] = lr_dict['random forest actual'] - lr_dict['random forest prediction']

        lr_dict['risk score'] = np.where(
            lr_dict['random forest actual'] > lr_dict['random forest prediction'],
            'under',
            'over'
        )
        print(lr_dict.head().to_string())
        lin_r2 = r2_score(y1, rg_pre)
        lin_mae = mean_absolute_error(y1, rg_pre)
        lin_mse = mean_squared_error(y1, rg_pre)
        lin_cvs = cross_val_score(en, feature, y1, cv=5, scoring='r2').mean()

        print('random forest r2 score: ', lin_r2)
        print('random forest mean absolute error: ', lin_mae)
        print('random forest mean squared error: ', lin_mse)
        print()
        print('random forest cross val score: ', lin_cvs)
        print()

        self.df1 = self.df1.join(lr_dict)
        self.df1.insert(0, 'id', self.df1.index+1)
        self.df1.reset_index(drop=True)
        print(self.df1.head().to_string())
        #self.df1.to_csv('c:/Users/anton/supervised/final2.csv', index=False)


if __name__ == "__main__":
    b = B('c:/Users/anton/unsuper/reg/raw_reg1.csv')

