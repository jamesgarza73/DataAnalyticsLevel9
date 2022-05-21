import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats
import pingouin as pg
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Stats():  
    def shapiroTest(data, col):
        stat = stats.shapiro(data)
        if stat[1] >= 0.05:
            print('{} normal distribution, pvalue = {:f}'.format(col, stat[1]))
            normal = True
        else:
            print('{} not normal distribution, pvalue = {:f}'.format(col, stat[1]))
            normal = False
        return normal
    
    def getArgs(data):
        cols={}
        for col in data.columns:
            cols[col] = 'data['+col+']'
        args = cols.values()
        return args  
    
    def oneWayAnova(data, input1, input2):
        #ONE-WAY ANOVA
        inputs = input1 + "~" + input2
        model = ols(inputs, data = data).fit()
        aov = sm.stats.anova_lm(model, type=2)
        return aov
    
    def twoWayAnova(data, input1, input2, input3):
        #ANOVA TWO WAYS
        inputs = input1 + '~' + input2 + '~' + input3
        model = ols(inputs, data = data).fit()
        aov = sm.stats.anova_lm(model, type=2)
        return aov
    
    def twoGroupCompare(data, results):
        cols = data.columns
        if all(results):
            # perform T-Test
            pValue = stats.ttest_ind(data[cols[0]], data[cols[1]])
            if pValue[1] >=0.05:
                print('Accept Null Hypothesis P-Value {:.4f}'.format(pValue[1]))
            else:
                print('Reject Null Hypothesis P-Value {:.4f}'.format(pValue[1])) 
            return pValue
        else:
            #perform the Mann-Whitney U test
            MW = stats.mannwhitneyu(data[cols[0]], data[cols[1]], 
               alternative='two-sided')
        if MW[1] >= 0.05:
            print('Accept Null Hypothesis P-Value {:.4f}'.format(MW[1]))
        else:
            print('Reject Null Hypothesis P-Value {:.4f}'.format(MW[1]))     
        return MW
    
    def groupCompare(data, results, input1, input2, input3):
        cols = data.columns
        if all(results):
            # ANOVA
            if not input3:
                oneWay = Stats.oneWayAnova(data, input1, input2)
                if oneWay[1] >=0.05:
                    print('Accept Null Hypothesis P-Value {:.4f}'.format(oneWay[1]))
                else:
                    print('Reject Null Hypothesis P-Value {:.4f}'.format(oneWay[1])) 
                return oneWay
            else:
                twoWay = Stats.twoWayAnova(data, input1, input2, input3)
                if twoWay[1] >=0.05:
                    print('Accept Null Hypothesis P-Value {:.4f}'.format(twoWay[1]))
                else:
                    print('Reject Null Hypothesis P-Value {:.4f}'.format(twoWay[1])) 
                return twoWay
        else:
            # Krustal Wallis
            args = Stats.getArgs(data)
            kurkWall = stats.kruskal(*args)
            if kurkWall[1] >=0.05:
                print('Accept Null Hypothesis P-Value {:.4f}'.format(kurkWall[1]))
            else:
                print('Reject Null Hypothesis P-Value {:.4f}'.format(kurkWall[1])) 
            return kurkWall
  

    def distribution(data, input1=None, input2=None, input3=None):
        results = []
        for col in data.columns:
            results.append(Stats.shapiroTest(data[col], col))
        if all(results):
            if data.shape[1] > 2:
                if not input3:
                    # Add Levene
                    # One Way Anova
                    test = Stats.groupCompare(data, results, input1, input2, input3=None)
                else:
                    # Two Way Anova
                    levene = pg.homoscedasticity(data, method='levene')
                    print(levene +'\n')
                    if levene['equal_var'].values[0]:
                        test = Stats.groupCompare(data, results, input1, input2, input3)
                    else:
                        print('Variances are not equal to perform two way ANOVA')
            else:
                # TTest
                test = Stats.twoGroupCompare(data, results)
        else: 
            if data.shape[1] > 2:
                # Kruskal-Wallis
                test = Stats.groupCompare(data, results, input1=None, input2=None, input3=None)
            else:
                # Mann-Whitney
                test = Stats.twoGroupCompare(data, results)
        return test

class dataProcessing():
    def __init__(self, data, target, rollingDays):
        self.data = data
        self.target = target
        self.targetData = None
        self.rollingDays = rollingDays
        self.dataNoOutliers = data
        
    def targetIntoLast(self):
        # Put target into last column
        self.targetData = self.data[self.target]
        self.data.drop(columns=[self.target], inplace=True)
        self.data[self.target] = self.targetData
        return self.data

    def movingAverage(self):
        if 'MovingAverage' not in self.data.columns:
            # Add moving Average if not in dataset
            self.data.insert(loc=self.data.shape[1] - 2, column='MovingAverage',
                            value=self.data[self.target].rolling(self.rollingDays).mean())
        else:
            # Change moving average
            self.data['MovingAverage'] = self.data[self.target].rolling(self.rollingDays).mean()
        return self.data
    
    def highLow(self):
        if 'High' not in self.data.columns and 'Low' not in self.data.columns:
            # Add High and Low if not in dataset
            self.data.insert(loc=self.data.shape[1] - 2, column='High',
                            value=self.data[self.target].rolling(self.rollingDays).max())
            self.data.insert(loc=self.data.shape[1] - 2, column='Low',
                            value=self.data[self.target].rolling(self.rollingDays).min())
        else:
            # Change High and Low 
            self.data['High'] = self.data[self.target].rolling(self.rollingDays).max()
            self.data['Low'] = self.data[self.target].rolling(self.rollingDays).min()
        return self.data
        
    def addMonth(self):
        # Add month
        if 'Month' not in self.data.columns:
            self.data.insert(loc=0, column='Month', value=self.data.index.month)
            self.data.Month = self.data.Month.astype(str)
        return self.data
    
    def removeOutliers(self):
        cols = self.data.select_dtypes(exclude=['object']).columns.tolist()
        for col in cols:
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            # No below zero values
            if lower < 1:
                lower = 1
            upper = q3 + 1.5 * iqr
            self.dataNoOutliers = self.dataNoOutliers.loc[(self.data[col] > lower) &
                                (self.data[col] < upper)]
        return self.dataNoOutliers
        
class Models():
    def runModels(self, regressors, parameters, addEstimators, scalers, X, y, outliers):
        results = pd.DataFrame(columns=['Model Name', 'Scaler', 'Outliers',
                                        'Dimension Reducer', 'Best Parameters', 'R2 Score',
                                       'Mean Square Error', 'Mean Absolute Error',
                                       'Feature Importance'])
        count = 1
        total = len(regressors) * len(addEstimators) * len(scalers)
        # https://lifewithdata.com/2022/03/09/onehotencoder-how-to-do-one-hot-encoding-in-sklearn/
        # get the categorical and numeric column names
        num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        for h, i in addEstimators.items():
            for l, m in scalers.items():
                # Create num columns for pipeline.
                num_pipe = make_pipeline(KNNImputer(n_neighbors=3), m)
                for j, k in regressors.items():
                    feature_importance = 'Not Available'
                    dimensionReduction = 'Not Available'
                    if j != 'PolynomialFeatures':
                        pipe = make_pipeline(ColumnTransformer([('cat', 
                                    OneHotEncoder(handle_unknown='ignore'), cat_cols),
                                    ('num', num_pipe, num_cols)]), i, k())

                        grid = GridSearchCV(pipe, param_grid=parameters[j]['gridParams'], cv=5)
                        grid.fit(X, y)
                        if j == 'XGBRegressor':
                            feature_importance = grid.best_estimator_.named_steps['xgbregressor'].feature_importances_.astype('object')
                        if h == 'PCA':
                            dimensionReduction = grid.best_estimator_.named_steps['pca'].components_
                        if h == 'KMeans':
                            dimensionReduction = Counter(grid.best_estimator_.named_steps['kmeans'].labels_)
                        results = results.append({'Model Name':  j,'Scaler': l,
                                        'Outliers' : outliers, 'Dimension Reducer': h, 'Dimension Reduction': dimensionReduction,
                                       'Best Parameters': str(grid.best_params_),
                                       'R2 Score': grid.score(X,y), 'Mean Square Error':
                                       mean_squared_error(y, grid.predict(X)), 
                            'Mean Absolute Error': mean_absolute_error(y, grid.predict(X)),
                                                  'Feature Importance': feature_importance}, 
                                                 ignore_index=True)
                    elif j == 'PolynomialFeatures':
                        pipe = make_pipeline(ColumnTransformer([('cat', 
                                OneHotEncoder(handle_unknown='ignore'), cat_cols),
                                ('num', num_pipe, num_cols)]), i, k(), BayesianRidge())
                        # Add parameters unique to 
                        grid = GridSearchCV(pipe, 
                            param_grid=parameters[j]['gridParams'], cv=5)
                        grid.fit(X, y)
                        if h == 'PCA':
                            dimensionReduction = grid.best_estimator_.named_steps['pca'].components_
                        if h == 'KMeans':
                            dimensionReduction = Counter(grid.best_estimator_.named_steps['kmeans'].labels_)
                        results = results.append({'Model Name':  'Polynomial Features Bayesian','Scaler': l,
                                        'Outliers' : outliers, 'Dimension Reducer': h, 'Dimension Reduction': dimensionReduction,
                                       'Best Parameters': str(grid.best_params_),
                                       'R2 Score': grid.score(X,y), 'Mean Square Error':
                                       mean_squared_error(y, grid.predict(X)), 
                            'Mean Absolute Error': mean_absolute_error(y, grid.predict(X)),
                                                  'Feature Importance': feature_importance}, 
                                                 ignore_index=True)
                    print('Model {}'.format(j))
                    print('Result for this model is {:2f} % '.format(grid.score(X,y)))
                    print('Completed {}% \n'.format(round((count / total)*100,2)))
                    count += 1
        return results