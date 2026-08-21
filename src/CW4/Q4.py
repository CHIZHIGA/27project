"""Question 4: ECG stationarity, decomposition and forecast comparison."""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "single_ecg_signal.csv"
RESULTS = ROOT / "results" / "CW4" / "Q4"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    sns.set_theme(style="whitegrid")
    signal_raw = pd.read_csv(DATA_PATH, header=None)[0]
    removed_sample = float(signal_raw.iloc[0])
    signal = signal_raw.iloc[1:].reset_index(drop=True).astype(float).rename("ECG")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(signal, lw=0.9)
    ax.set(
        title=f"Original ECG signal (first artefact value {removed_sample:g} removed)",
        xlabel="Sample index",
        ylabel="Amplitude",
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "original_signal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    transformed = {
        "Original": signal,
        "First difference": signal.diff().dropna(),
        "Square root": np.sqrt(signal - signal.min() + 1),
        "Log": np.log(signal - signal.min() + 1),
    }
    stationarity_rows = []
    for name, series in transformed.items():
        statistic, p_value, used_lag, n_obs, critical, _ = adfuller(series)
        stationarity_rows.append(
            {
                "transform": name,
                "ADF_statistic": statistic,
                "p_value": p_value,
                "critical_value_1pct": critical["1%"],
                "critical_value_5pct": critical["5%"],
                "critical_value_10pct": critical["10%"],
                "lags_used": used_lag,
                "observations": n_obs,
                "stationary_at_5pct": p_value < 0.05,
            }
        )
    stationarity = pd.DataFrame(stationarity_rows)
    stationarity.to_csv(RESULTS / "stationarity_tests.csv", index=False)
    pd.DataFrame({name: pd.Series(series) for name, series in transformed.items()}).to_csv(
        RESULTS / "transformed_signals.csv", index=False
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, name in zip(axes, ["First difference", "Square root", "Log"]):
        ax.plot(transformed[name], lw=0.8)
        ax.set(title=name, ylabel="Value")
    axes[-1].set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(RESULTS / "signal_transforms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # The original signal is already ADF-stationary, so keep its amplitude scale.
    split_index = 1800
    train = signal.iloc[:split_index]
    test = signal.iloc[split_index:]

    assumed_period = 200
    decomposition = seasonal_decompose(
        train, model="additive", period=assumed_period, extrapolate_trend="freq"
    )
    decomposition_table = pd.DataFrame(
        {
            "observed": decomposition.observed,
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
        }
    )
    decomposition_table.to_csv(RESULTS / "seasonal_decomposition.csv", index=True)
    figure = decomposition.plot()
    figure.set_size_inches(12, 8)
    figure.suptitle(
        f"Additive decomposition (assumed period = {assumed_period} samples)",
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(RESULTS / "seasonal_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_acf(train, lags=250, zero=False, ax=axes[0])
    plot_pacf(train, lags=40, zero=False, method="ywm", ax=axes[1])
    axes[0].set_title("Autocorrelation function")
    axes[1].set_title("Partial autocorrelation function")
    fig.tight_layout()
    fig.savefig(RESULTS / "acf_pacf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    candidate_orders = {
        "AR(3)": (3, 0, 0),
        "MA(2)": (0, 0, 2),
        "ARMA(3,2)": (3, 0, 2),
        "ARIMA(3,1,2)": (3, 1, 2),
    }
    train_values = train.to_numpy()
    test_values = test.to_numpy()
    forecasts = {}
    metric_rows = []

    for model_name, order in candidate_orders.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(train_values, order=order).fit()
        prediction = np.asarray(fitted.forecast(steps=len(test_values)))
        errors = test_values - prediction
        forecasts[model_name] = prediction
        metric_rows.append(
            {
                "model": model_name,
                "order": str(order),
                "RSS": np.sum(errors**2),
                "RMSE": np.sqrt(mean_squared_error(test_values, prediction)),
                "MAE": mean_absolute_error(test_values, prediction),
                "train_AIC": fitted.aic,
                "train_BIC": fitted.bic,
            }
        )

    mean_prediction = np.repeat(train.mean(), len(test_values))
    mean_errors = test_values - mean_prediction
    metric_rows.append(
        {
            "model": "Training-mean baseline",
            "order": "baseline",
            "RSS": np.sum(mean_errors**2),
            "RMSE": np.sqrt(mean_squared_error(test_values, mean_prediction)),
            "MAE": mean_absolute_error(test_values, mean_prediction),
            "train_AIC": np.nan,
            "train_BIC": np.nan,
        }
    )
    forecast_metrics = pd.DataFrame(metric_rows).sort_values("RSS")
    forecast_metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)

    test_index = np.arange(split_index, split_index + len(test_values))
    forecast_table = pd.DataFrame(
        {
            "sample_index": test_index,
            "observed": test_values,
            **forecasts,
            "Training-mean baseline": mean_prediction,
        }
    )
    forecast_table.to_csv(RESULTS / "forecasts.csv", index=False)

    requested = forecast_metrics[
        forecast_metrics["model"] != "Training-mean baseline"
    ]
    best_model = str(requested.iloc[0]["model"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(test_index, test_values, color="black", lw=0.8, label="Observed")
    for name, prediction in forecasts.items():
        axes[0].plot(test_index, prediction, lw=1.2, alpha=0.8, label=name)
    axes[0].set(
        title="Multi-step forecasts over the complete hold-out segment",
        ylabel="Amplitude",
    )
    axes[0].legend(ncol=3, fontsize=9)

    zoom = 300
    axes[1].plot(
        test_index[:zoom], test_values[:zoom], color="black", lw=1, label="Observed"
    )
    axes[1].plot(
        test_index[:zoom],
        forecasts[best_model][:zoom],
        color="tab:red",
        lw=1.5,
        label=f"Best requested model: {best_model}",
    )
    axes[1].set(
        title="First 300 hold-out samples",
        xlabel="Sample index",
        ylabel="Amplitude",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "raw_samples": len(signal_raw),
        "removed_first_sample": removed_sample,
        "modelled_samples": len(signal),
        "train_samples": len(train),
        "test_samples": len(test),
        "train_mean": float(train.mean()),
        "test_mean": float(test.mean()),
        "assumed_decomposition_period": assumed_period,
        "selected_transform": "Original",
        "best_requested_model_by_RSS": best_model,
        "best_requested_RSS": float(requested.iloc[0]["RSS"]),
        "best_requested_RMSE": float(requested.iloc[0]["RMSE"]),
        "best_requested_MAE": float(requested.iloc[0]["MAE"]),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Stationarity tests:\n", stationarity.to_string(index=False))
    print("\nForecast comparison:\n", forecast_metrics.to_string(index=False))
    print(f"Saved Q4 outputs to {RESULTS}")


if __name__ == "__main__":
    main()
