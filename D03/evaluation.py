import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_validate
import preprocessing


def evaluate_random_forest_regressor(data: pd.DataFrame) -> (float,) * 2:
    """
    Evaluate the RandomForestRegressor

    :param data: a pandas DataFrame
    :return: mean absolute error, mean absolute percentage error
    """
    x_train, x_test, y_train, y_test = preprocessing.run_preprocessing_pipeline(data)
    # initialize the model
    rfr = RandomForestRegressor(bootstrap=False, max_depth=16, max_features='log2', min_samples_leaf=4,
                                min_samples_split=10, n_estimators=1000, n_jobs=-1, random_state=42)
    # fit the train data into the model
    rfr.fit(x_train, y_train)
    # make predictions
    y_prediction = rfr.predict(x_test)
    # mean absolute error
    mae = round(mean_absolute_error(y_test, y_prediction), 4)
    # mean absolute percentage error
    mape = round(mean_absolute_percentage_error(y_test, y_prediction), 4)

    return mae, mape

def evaluate_hist_gradient_boosting_regressor(data: pd.DataFrame) -> (float,) * 2:
    """
    Evaluate the RandomForestRegressor

    :param data: a pandas DataFrame
    :return: mean absolute error, mean absolute percentage error
    """
    x_train, x_test, y_train, y_test = preprocessing.run_preprocessing_pipeline(data)
    # initialize the model
    hgbr = HistGradientBoostingRegressor(learning_rate=0.1, loss='poisson', max_depth=25,
                                         max_iter=500, min_samples_leaf=15)
    # cross validation
    for i in range(2, 11):
        cv = KFold(n_splits=i, random_state=42, shuffle=True)
        scores = cross_validate(hgbr, x_train, y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    print(scores["test_score"])
    print(np.mean(scores["test_score"]))
    print(np.std(scores["test_score"]))
    # fit the train data into the model
    hgbr.fit(x_train, y_train)
    # make predictions
    y_prediction = hgbr.predict(x_test)
    # mean absolute error
    mae = round(mean_absolute_error(y_test, y_prediction), 4)
    # mean absolute percentage error
    mape = round(mean_absolute_percentage_error(y_test, y_prediction), 4)

    return mae, mape


if __name__ == "__main__":
    # read csv
    listings = pd.read_csv("../data/listings.csv")
    # print the errors for each regressor
    mae_rfr, mape_rfr = evaluate_random_forest_regressor(listings)
    print(mae_rfr, mape_rfr)

    mae_hgbr, mape_hgbr = evaluate_hist_gradient_boosting_regressor(listings)
    print(mae_hgbr, mape_hgbr)

