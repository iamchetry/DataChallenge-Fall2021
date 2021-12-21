from chemml.datasets import load_organic_density
import pandas as pd

from sklearn.model_selection import train_test_split


def get_necessary_data():
    '''

    :return: Train & Test splits of data,  Scaled value of target Density, Mean & Standard Deviation of Target Density
    '''
    molecules, target, dragon_subset = load_organic_density()

    X_train, X_test, y_train, y_test = train_test_split(dragon_subset, target, test_size=0.25,
                                                        stratify=pd.qcut(target['density_Kg/m3'], 10).astype('str'),
                                                        random_state=42)

    X_train.index = range(len(X_train))
    X_test.index = range(len(X_test))
    y_train.index = range(len(y_train))
    y_test.index = range(len(y_test))

    target_mean = y_train.mean()[0]
    target_std = y_train.std()[0]

    y_train_scaled = (y_train - target_mean) / target_std
    y_test_scaled = (y_test - target_mean) / target_std

    return X_train, X_test, y_train, y_test, y_train_scaled, y_test_scaled, target_mean, target_std
