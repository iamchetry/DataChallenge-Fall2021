import pandas as pd
import pickle

from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from hyperopt import hp, fmin, tpe, STATUS_OK
from sklearn.model_selection import cross_val_score


def lasso_regression(X_train, y_train, X_test, y_test, y_train_scaled, target_mean, target_std):
    '''

    :param X_train: Input Feature data for Train
    :param y_train: Output feature for Train (Density)
    :param X_test: Input Feature data for Test
    :param y_test: Output feature for Test (Density)
    :param y_train_scaled: Scaled output for Train (Scaled Density)
    :param target_mean: Mean of output feature (Density)
    :param target_std: Standard Deviation of output feature (Density)
    :return: Dumps the Actual v/s Predicted Values and LASSO Coefficients in csv
    '''
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train.values, y_train_scaled.values)

    with open('model_objects/chemml_lasso.pickle', 'wb') as handle:
        pickle.dump(lasso_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    y_train_predicted = [(_ * target_std) + target_mean for _ in list(lasso_model.predict(X_train))]
    y_test_predicted = [(_ * target_std) + target_mean for _ in list(lasso_model.predict(X_test))]

    df_train_lasso = pd.concat([y_train, pd.DataFrame({'predicted_density': y_train_predicted})], ignore_index=False,
                               axis=1)
    df_test_lasso = pd.concat([y_test, pd.DataFrame({'predicted_density': y_test_predicted})], ignore_index=False,
                              axis=1)

    df_train_lasso.to_csv('data/df_train_actual_vs_predicted_lasso.csv', index=False)
    df_test_lasso.to_csv('data/df_test_actual_vs_predicted_lasso.csv', index=False)

    df_lasso_coeffs = pd.DataFrame({'feature': list(X_train.columns), 'coefficients': list(lasso_model.coef_)})
    df_lasso_coeffs['abs_'] = df_lasso_coeffs.coefficients.abs()
    df_lasso_coeffs.sort_values(by='abs_', ascending=False, inplace=True)
    df_lasso_coeffs.index = range(len(df_lasso_coeffs))
    df_lasso_coeffs.drop(columns='abs_', axis=1, inplace=True)
    df_lasso_coeffs.to_csv('data/df_lasso_coefficients.csv', index=False)


def objective_fn_rfr(params, X_train, y_train_scaled):
    '''

    :param params: Hyper-parameter Grid
    :param X_train: Input Feature data for Train
    :param y_train_scaled: Scaled output for Train (Scaled Density)
    :return: Score value
    '''
    global it_, scores_
    params = {
        'max_depth': int(params['max_depth']),
        'n_estimators': int(params['n_estimators']),
        'min_samples_split': params['min_samples_split'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'],
        'oob_score': params['oob_score'],
        'max_samples': params['max_samples']
    }

    clf = RFR(n_jobs=3, **params)

    it_ = it_ + 1
    score = cross_val_score(clf, X_train.values, y_train_scaled.values.ravel(), scoring='neg_root_mean_squared_error',
                            cv=4).mean()

    with open("logs_rf.txt", "a") as myfile:
        myfile.write('------------------- On {} ------------------\n'.format(it_))
        myfile.write('Params : {}\n'.format(params))
        myfile.write('RMSE : {}\n'.format(-score))

    return {'loss': 1 - score, 'status': STATUS_OK}


def objective_fn_xgb(params, X_train, y_train_scaled):
    '''

    :param params: Hyper-parameter Grid
    :param X_train: Input Feature data for Train
    :param y_train_scaled: Scaled output for Train (Scaled Density)
    :return: Score value
    '''
    global it_, scores_
    params = {
        'max_depth': int(params['max_depth']),
        'n_estimators': int(params['n_estimators']),
        'reg_alpha': params['reg_alpha'],
        'reg_lambda': params['reg_lambda']
    }

    clf = XGBRegressor(learning_rate=0.01, n_jobs=3, **params)

    it_ = it_ + 1
    score = cross_val_score(clf, X_train.values, y_train_scaled.values, scoring='neg_root_mean_squared_error',
                            cv=4).mean()

    with open("logs_xgb.txt", "a") as myfile:
        myfile.write('------------------- On {} ------------------\n'.format(it_))
        myfile.write('Params : {}\n'.format(params))
        myfile.write('RMSE : {}\n'.format(-score))

    return {'loss': 1 - score, 'status': STATUS_OK}


def xgb_model():
    '''

    :return: Minimized Loss Function
    '''
    space = {
        'max_depth': hp.choice('max_depth', [5, 7, 10]),
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
        'reg_alpha': hp.choice('reg_alpha', [0.01, 0.1, 0.5, 1]),
        'reg_lambda': hp.choice('reg_lambda', [0.01, 0.1, 0.5, 1])}

    return fmin(fn=objective_fn_xgb, space=space, algo=tpe.suggest, max_evals=800)


def random_forest_model():
    '''

    :return: Minimized Loss Function
    '''
    space = {
        'max_depth': hp.choice('max_depth', [5, 7, 10]),
        'n_estimators': hp.choice('n_estimators', [50, 125, 200]),
        'min_samples_split': hp.choice('min_samples_split', [2, 4, 6]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5]),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
        'oob_score': hp.choice('oob_score', [True, False]),
        'max_samples': hp.choice('max_samples', [100, 150, 200])}

    return fmin(fn=objective_fn_rfr, space=space, algo=tpe.suggest, max_evals=1500)


def dump_xgboost_model(X_train, y_train_scaled):
    '''

    :param X_train: Input Feature data for Train
    :param y_train_scaled: Scaled output for Train (Scaled Density)
    :return: Dumps XGBOOST trained model
    '''
    params = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200, 'reg_alpha': 0.01, 'reg_lambda': 0.01}
    model_ = XGBRegressor(**params)
    model_.fit(X_train, y_train_scaled, verbose=True)
    with open('model_objects/chemml_xgboost.pickle', 'wb') as handle:
        pickle.dump(model_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dump_random_forest_model(X_train, y_train_scaled):
    '''

    :param X_train: Input Feature data for Train
    :param y_train_scaled: Scaled output for Train (Scaled Density)
    :return: Dumps Random Forest trained model
    '''
    params = {'max_depth': 10, 'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 1,
              'max_features': 'auto', 'oob_score': True, 'max_samples': 200}
    model_ = RFR(**params)
    model_.fit(X_train.values, y_train_scaled.values.ravel())
    with open('model_objects/chemml_rfr.pickle', 'wb') as handle:
        pickle.dump(model_, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_and_predict(model_object_path, target_mean, target_std, X_train, X_test, y_train, y_test):
    '''

    :param model_object_path: Path to model object pickle
    :param target_mean: Mean of output feature (Density)
    :param target_std: Standard Deviation of output feature (Density)
    :param X_train: Input Feature data for Train
    :param X_test: Input Feature data for Test
    :param y_train: Output feature for Train (Density)
    :param y_test: Output feature for Test (Density)
    :return: Actual v/s Predicted values for Train & Test
    '''
    with open(model_object_path, 'rb') as input_file:
        model_obj = pickle.load(input_file)

    y_train_predicted = [(_ * target_std) + target_mean for _ in list(model_obj.predict(X_train))]
    y_test_predicted = [(_ * target_std) + target_mean for _ in list(model_obj.predict(X_test))]

    df_train_xgb = pd.concat([y_train, pd.DataFrame({'predicted_density': y_train_predicted})], ignore_index=False,
                             axis=1)
    df_test_xgb = pd.concat([y_test, pd.DataFrame({'predicted_density': y_test_predicted})], ignore_index=False, axis=1)

    return df_train_xgb, df_test_xgb


def get_feature_importance(model_object_path, X_train):
    '''

    :param model_object_path: Path to model object pickle
    :param X_train: Input Feature data for Train
    :return: Feature Importance Data-Frame
    '''
    with open(model_object_path, 'rb') as input_file:
        model_obj = pickle.load(input_file)

    df_features = pd.DataFrame({'feature': list(X_train.columns), 'importance': list(model_obj.feature_importances_)})
    df_features.sort_values(by='importance', ascending=False, inplace=True)
    df_features.index = range(len(df_features))
    df_features.importance = round((df_features.importance / df_features.importance.sum()) * 100, 2)

    return df_features
