# import libraries
import argparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from features import transform_data
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to data")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--registered_model_name", type=str, default=None)

    return parser.parse_args()


def main():
    # for model flavors that support autologging
    mlflow.autolog()

    # parse arguments
    args = parse_args()

    # prepare data
    X_train, y_train, X_test, y_test = prepare_data(args)

    # train model
    model = train(args, X_train, y_train)

    # evaluate model
    eval(model, X_test, y_test)

    # save model
    save_model(args, X_test, model)


def prepare_data(args):
    print("Loading Data...")
    data_path = args.data
    df = pd.read_parquet(data_path)

    # process data
    df = transform_data(args, df)

    # split data
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # define target variable for training
    target = "targe_variable"

    X_train = train.drop([target], axis=1)
    y_train = train[target]

    X_test = test.drop([target], axis=1)
    y_test = test[target]

    return X_train, y_train, X_test, y_test


def train(args, X_train, y_train):
    model = HistGradientBoostingClassifier(
        learning_rate=args.lr,
        max_iter=args.ne,
        max_depth=args.max_depth,
    )

    print("Training Model...")
    model.fit(X_train, y_train)

    return model


def eval(model, X_test, y_test):
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, index=y_test.index)

    # log metrics
    # NOTE: while autologging may log some metrics, we need to ensure that any metrics
    # that are used as primary metrics for a sweep is logged manually
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", rmse)

    r2 = model.score(X_test, y_test)
    mlflow.log_metric("r2", r2)

    # log figure
    fig = plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    mlflow.log_figure(fig, "feature_importance.png")


def save_model(args, X_test, model):
    print("Saving Model...")

    # NOTE: some flavors do not support autologging, so we need to manually log the model

    # manual logging of model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_test.head(1),
        registered_model_name=args.registered_model_name,
    )


if __name__ == "__main__":
    main()
