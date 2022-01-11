import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def drop_unnecessary_elements(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the columns that are full of null values or contain urls, ids, text descriptions and dates
    These columns are the following:
    "neighbourhood_group_cleansed", "bathrooms", "calendar_updated",
    "listing_url", "picture_url", "host_url",  "host_thumbnail_url", "host_picture_url",
    "id", "scrape_id", "host_id",
    "name", "description", "neighborhood_overview", "host_name", "host_about", "host_location",
    "host_neighbourhood", "host_verifications", "neighbourhood", "license",
    "last_scraped", "calendar_last_scraped"

    :param data: a DataFrame
    :return: the same DataFrame after we have dropped the aforementioned columns
    """
    data_copy = data.copy()

    # drop the columns that are consisted of null values
    data_copy.drop(["neighbourhood_group_cleansed", "bathrooms", "calendar_updated"],
                   axis=1, inplace=True)

    # drop the columns that are consisted of urls
    data_copy.drop(["listing_url", "picture_url", "host_url", "host_thumbnail_url", "host_picture_url"],
                   axis=1, inplace=True)

    # drop the columns that are consisted of ids
    data_copy.drop(["id", "scrape_id", "host_id"],
                   axis=1, inplace=True)

    # drop the columns that are consisted of text descriptions
    data_copy.drop(["name", "description", "neighborhood_overview", "host_name", "host_about", "host_location",
                    "host_neighbourhood", "host_verifications", "neighbourhood", "license"],
                   axis=1, inplace=True)

    # drop some of the columns that are consisted of dates
    data_copy.drop(["last_scraped", "calendar_last_scraped"],
                   axis=1, inplace=True)

    # drop duplicates
    data_copy.drop_duplicates(inplace=True)

    # now rows with a lot of nan values can be found in the data set
    data_copy = data_copy.dropna(thresh=data_copy.shape[1] - 3)

    return data_copy


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the nan values of useful features with their respective median value

    :param data: a DataFrame
    :return: the same DataFrame after we have handled the missing values of significant rows
    """
    data_copy = data.copy()

    data_copy["bedrooms"].fillna(value=data_copy["bedrooms"].median(), inplace=True)
    data_copy["beds"].fillna(value=data_copy["beds"].median(), inplace=True)
    data_copy["reviews_per_month"].fillna(value=data_copy["reviews_per_month"].median(), inplace=True)
    data_copy["host_listings_count"].fillna(value=data_copy["host_listings_count"].median(), inplace=True)
    data_copy["host_total_listings_count"].fillna(value=data_copy["host_total_listings_count"].median(), inplace=True)

    # handle the missing values of date data
    # the procedure goes as follows ...
    # for each date and for each neighbourhood:
    # consider the subset of the dataframe where this specific neighbourhood tag appears in "neighbourhood_cleansed"
    # calculate the median value of the date in this subset
    # fill the nan values of this date with the previous median value in the specific rows.
    date_features_nan = ["host_since", "first_review", "last_review"]
    for date in date_features_nan:
        for neighbourhood in data_copy["neighbourhood_cleansed"].unique():
            condition = data_copy["neighbourhood_cleansed"] == neighbourhood
            # convert the object type into datetime64[ns] and get the median
            neighbourhood_date_series = data_copy[condition][date]
            neighbourhood_date_series = pd.to_datetime(neighbourhood_date_series)
            median = neighbourhood_date_series.quantile(0.5, interpolation="midpoint")
            # fill the specific nan values
            data_copy[date] = data_copy[date].mask(condition, data_copy[date].fillna(value=str(median)[0:10]))

    return data_copy


def strings_to_digits(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the object types of some useful columns into numerical types

    :param data: a DataFrame
    :return: the same DataFrame after we have converted strings characters into numbers
    """
    data_copy = data.copy()

    # create two features for each date feature which consist of the respective year and season
    # 0 stands for spring
    # 1 stands for summer
    # 2 stands for autumn
    # 3 stands for winter
    date_features = ["host_since", "first_review", "last_review"]
    for date in date_features:
        if data_copy[date].isnull().sum() > 0:
            raise ValueError("handle the missing values first")
        years = []
        seasons = []
        for i in range(len(data_copy)):
            years.append(int(data_copy[date].iloc[i][0:4]))
            month = data_copy[date].iloc[i][5:7]
            if month == "03" or month == "04" or month == "05":
                seasons.append(0)
            elif month == "06" or month == "07" or month == "08":
                seasons.append(1)
            elif month == "09" or month == "10" or month == "11":
                seasons.append(2)
            else:
                seasons.append(3)
        # construct the new feature
        data_copy[date + "_year"] = years
        data_copy[date + "_season"] = seasons
        # drop the old one
        data_copy.drop(date, axis=1, inplace=True)

    # convert the bathrooms text into number
    data_copy['bathrooms_text'] = data_copy['bathrooms_text'].str[0]
    data_copy['bathrooms'] = pd.to_numeric(data_copy['bathrooms_text'], errors='coerce')
    data_copy["bathrooms"].fillna(value=data_copy["bathrooms"].median(), inplace=True)
    data_copy.drop("bathrooms_text", axis=1, inplace=True)

    # convert "host_response_rate" and "host_acceptance_rate" into float
    data_copy['host_response_rate'] = data_copy["host_response_rate"].str.rstrip('%').astype('float') / 100.0
    data_copy["host_response_rate"].fillna(value=data_copy["host_response_rate"].median(), inplace=True)

    data_copy['host_acceptance_rate'] = data_copy["host_acceptance_rate"].str.rstrip('%').astype('float') / 100.0
    data_copy["host_acceptance_rate"].fillna(value=data_copy["host_acceptance_rate"].median(), inplace=True)

    return data_copy


def target_label(data: pd.DataFrame) -> pd.DataFrame:
    """
    Change the type of the target label

    :param data: a pandas DataFrame
    :return: the same DataFrame
    """
    data_copy = data.copy()

    # convert the target label into float type
    data_copy["price"] = [float(x[1:].replace(",", "")) if "," in x else float(x[1:]) for x in data_copy["price"]]

    return data_copy


def encode_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode the following features:
    "host_response_time", "host_is_superhost", "room_type", "host_has_profile_pic", "host_identity_verified",
    "has_availability", "instant_bookable", "neighbourhood_cleansed", "property_type"
    One-hot encode the following feature: "amenities"

    :param data: a DataFrame
    :return: the encoded DataFrame
    """
    data_copy = data.copy()

    # define the label-encoding maps
    host_response_time_map = {np.nan: 0, "within an hour": 0, "within a few hours": 1,
                              "within a day": 2, "a few days or more": 3}
    host_is_superhost_map = {np.nan: 0, "f": 0, "t": 1}
    host_identity_verified_map = {"f": 0, np.nan: 1, "t": 1}
    room_type_map = {"Entire home/apt": 0, "Private room": 1, "Shared room": 2, "Hotel room": 3}
    has_availability_map = {"f": 0, "t": 1}
    host_has_profile_pic_map = {"f": 0, np.nan: 1, "t": 1}
    instant_bookable_map = {"f": 0, "t": 1}

    # define the frequency-encoding maps
    neighbourhood_cleansed_map = dict(round(data_copy["neighbourhood_cleansed"].value_counts() / len(data_copy), 3))
    property_type_map = dict(round(data_copy["property_type"].value_counts() / len(data_copy), 3))

    # label-encode the variables
    data_copy["host_response_time"] = data_copy["host_response_time"].map(host_response_time_map)
    data_copy["host_is_superhost"] = data_copy["host_is_superhost"].map(host_is_superhost_map)
    data_copy["room_type"] = data_copy["room_type"].map(room_type_map)
    data_copy["host_identity_verified"] = data_copy["host_identity_verified"].map(host_identity_verified_map)
    data_copy['host_has_profile_pic'] = data_copy['host_has_profile_pic'].map(host_has_profile_pic_map)
    data_copy["has_availability"] = data_copy["has_availability"].map(has_availability_map)
    data_copy["instant_bookable"] = data_copy["instant_bookable"].map(instant_bookable_map)

    data_copy["neighbourhood_cleansed"] = data_copy["neighbourhood_cleansed"].map(neighbourhood_cleansed_map)
    data_copy["property_type"] = data_copy["property_type"].map(property_type_map)

    # one-hot encode the "amenities" variable
    top_20_amenities = {"Essentials": [], "Wifi": [], "Hair dryer": [], "Long term stays allowed": [],
                        "Air conditioning": [], "Hangers": [], "Kitchen": [], "Iron": [],
                        "Heating": [], "Hot water": [], "Dishes and silverware": [], "TV": [],
                        "Cooking basics": [], "Refrigerator": [], "Coffee maker": [], "Dedicated workspace": [],
                        "Shampo": [], "Bed linens": [], "Elevator": [], "Fire extinguisher": []}

    # for each sample and for each amenity of the previous dictionary:
    # if amenity exists in "amenities", then we give the value 1 (it exists).
    # Otherwise, we give the value 0 (it does not exist).
    for idx in range(len(data_copy)):
        amenities_idx = data_copy["amenities"].iloc[idx].split(", ")
        for idy in range(len(amenities_idx)):
            if "[" in amenities_idx:
                amenities_idx[idy] = amenities_idx[idy][2:]
            if "]" in amenities_idx:
                amenities_idx[idy] = amenities_idx[idy][:-2]
            amenities_idx[idy] = amenities_idx[idy][1:-1]
            if idy == 0:
                amenities_idx[idy] = amenities_idx[idy][1:]
            if idy == len(amenities_idx) - 1:
                amenities_idx[idy] = amenities_idx[idy][:-2]
        for amenity in top_20_amenities.keys():
            if amenity in amenities_idx:
                top_20_amenities[amenity].append(1)
            else:
                top_20_amenities[amenity].append(0)
    for amenity in top_20_amenities.keys():
        data_copy["amenity_" + amenity.lower()] = top_20_amenities[amenity]
    data_copy.drop("amenities", axis=1, inplace=True)

    return data_copy


def prepare_dataframe_for_ml(data: pd.DataFrame, test_size: float = 0.3) -> (np.ndarray,) * 4:
    """
    Takes an encoded DataFrame and splits it into training and test sets.

    :param data: A pandas DataFrame
    :param test_size: Ratio of data to be used for test
    :return: four arrays (train features, test features, train labels, test labels)
    """

    X = data.drop('price', axis=1)
    y = data['price']

    return train_test_split(X, y, test_size=test_size, random_state=42)


def scale_features(x_train: np.ndarray, x_test: np.ndarray, how: str = 'standardize') -> (np.ndarray, np.ndarray):
    """
    Scale features in train and test sets

    :param x_train: training features
    :param x_test: test features
    :param how: how to scale features (either 'normalize' or 'standardize')
    :return: the scaled training and test features
    """
    if how == 'normalize':
        scaler = MinMaxScaler()
    elif how == 'standardize':
        scaler = StandardScaler()
    else:
        raise ValueError('invalid value for parameter "how". Please select one of ("normalize", "standardize")')

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # save the scaler in a binary form
    joblib.dump(scaler, open("fastapi/data/scaler.joblib", "wb"))

    return x_train_scaled, x_test_scaled


def run_preprocessing_pipeline(data: pd.DataFrame) -> (np.ndarray,) * 4:
    """
    Run full preprocessing pipeline on input DataFrame.

    :param data: Input DataFrame
    :return: preprocessed train/test sets
    """

    data = drop_unnecessary_elements(data)
    data = handle_missing_values(data)
    data = strings_to_digits(data)
    data = target_label(data)
    data = encode_variables(data)

    x_train, x_test, y_train, y_test = prepare_dataframe_for_ml(data)

    # handle outliers on train data
    non_outlier_mask = (np.abs(stats.zscore(x_train)) < 7).all(axis=1)
    x_train = x_train[non_outlier_mask]
    y_train = y_train[non_outlier_mask]

    # scale features
    x_train, x_test = scale_features(x_train, x_test)

    return x_train, x_test, y_train, y_test

