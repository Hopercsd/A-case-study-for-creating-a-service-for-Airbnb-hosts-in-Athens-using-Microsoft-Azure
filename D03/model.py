import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import preprocessing


def get_regressor(data: pd.DataFrame) -> HistGradientBoostingRegressor:
    """
    Trains a StackingRegressor model and returns it

    :param data: a pandas DataFrame
    :return: regression model
    """
    # run preprocessing pipeline and get train and test sets
    x_train, x_test, y_train, y_test = preprocessing.run_preprocessing_pipeline(data)
    # initialize the regressor model
    hgbr = HistGradientBoostingRegressor(learning_rate=0.1, loss='poisson', max_depth=25,
                                         max_iter=500, min_samples_leaf=15)
    # fit the train data into the model
    hgbr.fit(x_train, y_train)

    return hgbr


if __name__ == "__main__":
    # read csv
    listings = pd.read_csv("../data/listings.csv")

    # stacking regressor
    regressor_model = get_regressor(listings)

    # save the model in a binary form
    joblib.dump(regressor_model, open("fastapi/data/regressor.joblib", "wb"))

