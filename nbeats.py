import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, mape, mse, rmse, r2_score
from typing import Optional, Union


def read_series(csv_filename: str) -> TimeSeries:
    """
    Read the data.

    :param csv_filenam (str): The input CSV file name
    :return: A 'TimeSeries' object of the specified 'server'
    """
    data = pd.read_csv(csv_filename)
    series = TimeSeries.from_series(data).astype(np.float32)
    return series


def define_NBEATS_model(
    train_set: Optional[TimeSeries] = None,
    val_set: Optional[TimeSeries] = None,
    gridsearch: bool = False,
) -> NBEATSModel:
    """
    Setup N-Beats model's architecture.

    :param train_set (Optional[TimeSeries]): Train set (used in grid search)
    :param val_set (Optional[TimeSeries]): Validation set (used in grid search)
    :param gridsearch (Optional[bool]): Perform grid search or not
    :return: An N-Beats model
    """
    if gridsearch == True:
        parameters = {
            "input_chunk_length": [16, 32],
            "output_chunk_length": [1],
            "num_stacks": [16, 30],
            "num_blocks": [1, 2, 3, 5, 10],
            "num_layers": [2, 3, 4],
            "layer_widths": [256, 512, 1024],
            "n_epochs": [20],
            "nr_epochs_val_period": [1],
            "batch_size": [128, 256, 512, 1024],
            "model_name": ["nbeats_run"],
        }

        # Randomized gridsearch
        res = NBEATSModel.gridsearch(
            parameters=parameters,
            series=train_set,
            val_series=val_set,
            start=0.1,  # starting point in training set
            last_points_only=False,
            metric=mape,
            reduction=np.mean,
            n_jobs=-1,
            n_random_samples=0.99,  # % of full search space to evaluate
            verbose=True,
        )
        best_model, dict_bestparams = res
        print(f"dict_bestparams: {dict_bestparams}")
        model_nbeats = best_model
    else:
        # Generic architecture
        model_nbeats = NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=1,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=4,
            num_layers=4,
            layer_widths=512,
            n_epochs=100,
            nr_epochs_val_period=1,
            batch_size=800,
            model_name="nbeats_run",
        )
    return model_nbeats


def display_forecast(
    pred_series: TimeSeries,
    observed_series: TimeSeries,
    forecast_type: str,
    start_date: Optional[Union[pd.Timestamp, float, int]] = None,
) -> None:
    """
    Plot observed versus predicted data.

    :param pred_series (TimeSeries): The predicted series
    :param observed_series (TimeSeries): The observed series
    :param forecast_type (str): Horizon, e.g. 7 days
    :param start_date (Optional[Union[Timestamp, float, int]]): Start of 'observed_series'
    :return: None
    """
    plt.figure(figsize=(8, 5))
    if start_date:
        observed_series = observed_series.drop_before(start_date)
    observed_series.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "MAPE: {}".format(mape(observed_series.univariate_component(0), pred_series))
    )
    plt.legend()

    MSE = mse(observed_series.univariate_component(0), pred_series)
    RMSE = rmse(observed_series.univariate_component(0), pred_series)
    MAE = mae(observed_series.univariate_component(0), pred_series)
    MAPE = mape(observed_series.univariate_component(0), pred_series)
    R2 = r2_score(observed_series.univariate_component(0), pred_series)
    # std = np.std(observed_series.univariate_component(0))

    print(f" MSE: {MSE}\n RMSE: {RMSE}\n MAE: {MAE}\n MAPE: {MAPE}\n R2: {R2}")


if __name__ == "__main__":
    series = read_series("data.csv")

    trainset_size = 0.6
    train, val = series.split_after(trainset_size)

    model_nbeats = define_NBEATS_model(train_set=train, val_set=val, gridsearch=True)
    model_nbeats.fit(series=train, val_series=val, verbose=True)

    pred_series = model_nbeats.historical_forecasts(
        series,
        start=trainset_size,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True,
    )
    display_forecast(pred_series, series, "1 horizon", start_date=trainset_size)
