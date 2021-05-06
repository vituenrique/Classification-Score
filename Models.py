from numpy import mean, std
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold
from xgboost import XGBRFRegressor, XGBRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

PATH = "..."

ROC_RESULTADOS = {
    'XGBRF': [],
    'XGB': [],
    'RF': [],
    'MLP': [],
    'SGD': []
}

N_TRIES = 5
SIZE_SAMPLE = 100000

def XGBRF(xTrain, yTrain, xTest, yTest):

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7],
        'reg_alpha': [0., 0.2, 0.5, 1., 2.],
        'reg_lambda': [0., 0.2, 0.5, 1., 2.],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'n_estimators': [100, 300, 600, 800],
        'objective': ['reg:logistic', 'binary:logistic']
    }

    reg = XGBRFRegressor()

    random_search = RandomizedSearchCV(reg, 
                                    param_distributions=params,
                                    n_iter=30, 
                                    scoring='roc_auc',
                                    n_jobs=2,
                                    verbose=2)

    random_search.fit(
        xTrain,
        yTrain
    )

    modelXGBRF = random_search.best_estimator_

    y_pred = modelXGBRF.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    roc = roc_auc_score(yTest, y_pred)

    ROC_RESULTADOS['XGBRF'].append(roc)

    print('XGBRF Mean Accuracy: ' + str(mse) + ' | ' + str(roc))

def XGB(xTrain, yTrain, xTest, yTest):

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7],
        'reg_alpha': [0., 0.2, 0.5, 1., 2.],
        'reg_lambda': [0., 0.2, 0.5, 1., 2.],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'n_estimators': [100, 300, 600, 800],
        'objective': ['reg:logistic', 'binary:logistic']
    }

    reg = XGBRegressor()

    random_search = RandomizedSearchCV(reg, 
                                    param_distributions=params,
                                    n_iter=30, 
                                    scoring='roc_auc',
                                    n_jobs=2,
                                    verbose=2)

    random_search.fit(
        xTrain,
        yTrain
    )

    modelXGB = random_search.best_estimator_

    y_pred = modelXGB.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    roc = roc_auc_score(yTest, y_pred)

    ROC_RESULTADOS['XGB'].append(roc)

    print('XGB Mean Accuracy: ' + str(mse) + ' | ' + str(roc))

def RF(xTrain, yTrain, xTest, yTest):
    modelRF = RandomForestRegressor(bootstrap=True,
                                max_features=0.2,
                                min_samples_leaf=18,
                                min_samples_split=20,
                                n_estimators=500,
                                n_jobs=4)
    modelRF.fit(xTrain, yTrain)

    y_pred = modelRF.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    roc = roc_auc_score(yTest, y_pred)

    ROC_RESULTADOS['RF'].append(roc)

    print('RF Mean Accuracy: ' + str(mse) + ' | ' + str(roc))

def MLP(xTrain, yTrain, xTest, yTest):
    modelMLP = MLPRegressor(random_state=1, max_iter=200, solver='sgd', activation='logistic')
    modelMLP.fit(xTrain, yTrain)

    y_pred = modelMLP.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    roc = roc_auc_score(yTest, y_pred)

    ROC_RESULTADOS['MLP'].append(roc)

    print('MLP Mean Accuracy: ' + str(mse) + ' | ' + str(roc))

def SGD(xTrain, yTrain, xTest, yTest):
    modelSGD = SGDRegressor(random_state=1, max_iter=500)
    modelSGD.fit(xTrain, yTrain)

    y_pred = modelSGD.predict(xTest)
    mse = mean_squared_error(yTest, y_pred)
    roc = roc_auc_score(yTest, y_pred)

    ROC_RESULTADOS['SGD'].append(roc)

    print('SGD Mean Accuracy: ' + str(mse) + ' | ' + str(roc))

if __name__ == '__main__':
    for i in range(N_TRIES):
        base = pd.read_pickle(PATH)
        base = base.sample(n=SIZE_SAMPLE)

        base = base.sort_values(['X4'], ascending=True).reset_index(drop=True) 

        flag_columns = base.columns[pd.Series(base.columns).str.startswith('Flag_')]

        for fc in flag_columns:
            base[fc] = base[fc].astype(int)

        X_train = base[base.X4 < base.X4.quantile(0.8)].drop(columns=['X8', 'X1', 'X2', 'X4']) * 1
        y_train = base[base.X4 < base.X4.quantile(0.8)].X8 * 1
        X_test = base[base.X4 >= base.X4.quantile(0.8)].drop(columns=['X8', 'X1', 'X2', 'X4']) * 1
        y_test = base[base.X4 >= base.X4.quantile(0.8)].X8 * 1

        XGBRF(X_train, y_train, X_test, y_test)
        XGB(X_train, y_train, X_test, y_test)
        RF(X_train, y_train, X_test, y_test)
        MLP(X_train, y_train, X_test, y_test)
        SGD(X_train, y_train, X_test, y_test)

    print(ROC_RESULTADOS)
    print('XGBRF: ' + str(mean(ROC_RESULTADOS['XGBRF'])) + ' | ' + str(std(ROC_RESULTADOS['XGBRF'])))
    print('XGB: ' + str(mean(ROC_RESULTADOS['XGB'])) + ' | ' + str(std(ROC_RESULTADOS['XGB'])))
    print('RF: ' + str(mean(ROC_RESULTADOS['RF'])) + ' | ' + str(std(ROC_RESULTADOS['RF'])))
    print('MLP: ' + str(mean(ROC_RESULTADOS['MLP'])) + ' | ' + str(std(ROC_RESULTADOS['MLP'])))
    print('SGD: ' + str(mean(ROC_RESULTADOS['SGD'])) + ' | ' + str(std(ROC_RESULTADOS['SGD'])))