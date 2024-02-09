"""
This script contains functions for feature engineering.

For ease of reading, please refrain from using more than 1 function per features transformation.
"""

# NOTE this is a really bad preprocessing function, it is just an example
def remove_outliers(df, lower_quantile, upper_quantile, column):
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)

    return df[(df[column] > lower_bound) & (df[column] < upper_bound)]


def string_to_category(df):
    df["VesselType"] = df.VesselType.astype("category")

    return df


def drop_na(df):
    return df.dropna()


def transform_data(args, df):

    df = drop_na(df)

    df = remove_outliers(df, args.lower_quantile, args.upper_quantile, args)

    df = string_to_category(df)

    return df
