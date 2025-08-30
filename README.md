# Model Performance Evaluation

This document breaks down the performance of the forecasting model based on the latest test results.

---

## 📊 Key Metrics Summary

| Metric                 | Value    | Quick Interpretation                       |
| ---------------------- | -------- | ------------------------------------------ |
| `RMSE`                 | 1.14     | Typical error is 1.14 percentage points.   |
| `MAE`                  | 0.91     | Average error is 0.91 percentage points.   |
| `Directional Accuracy` | 50.0%    | 🔴 **No better than a coin flip.**         |
| `R-squared (R²)`       | -0.13    | 🔴 **Worse than a naive "average" model.** |
| `SMAPE`                | 139.3%   | Very high percentage error.                |
| `MAPE`                 | 351,298% | (Ignore - distorted value)                 |

---

## 🎯 Detailed Analysis

### 1. How Big is the Average Mistake?

These first numbers tell you how far off your predictions are, on average.

#### `RMSE`: 1.14

The "typical" error. A value of 1.14 means your model's prediction for the percentage price change is usually wrong by about **1.14 percentage points**.

> **Analogy:** If the price actually went up by 2%, your model might have predicted it would go up by 0.86% or maybe 3.14%. It gives you the standard "margin of error" for its guesses.

#### `MAE`: 0.91

The simplest measure of error. It means that if you take every prediction, see how wrong it was, and average all those mistakes, the average mistake is **0.91 percentage points**.

> **Analogy:** This is the most honest, straightforward "average mistake" your model makes each day.

### 2. Does the Model Actually Have Skill?

These next numbers tell you if the model has learned anything useful.

#### `Directional Accuracy`: 50.0%

This answers the most important question: "Did the model correctly guess if the price would go UP or DOWN?" A score of 50% means it guessed right exactly half the time.

> **The Coin Flip Test:** Your model is as good as flipping a coin. This tells you the model has **no real predictive skill** right now.

#### `R-squared (R²)`: -0.13

This tells you how much better your model is than a "dumb" model that just predicts the average price change every single day. A perfect score is +1. A score of 0 means your model is equally as good as the dumb model.

> **The "Worse Than Nothing" Test:** Your score is negative. This is a major red flag. It means your model is **actively worse** than the simple, dumb model. This confirms the model isn't just unskilled; it's making actively poor predictions.

### 3. How Bad are the Errors in Percentage Terms?

These last numbers try to put the error into a percentage context, but they can be misleading.

#### `SMAPE`: 139.3%

This is a smarter version of MAPE that tries to fix the "divide-by-zero" problem. It's on a scale from 0% (perfect) to 200% (terrible).

> **Interpretation:** Your score of 139.3% is very high. It confirms that your model's prediction errors are very large compared to the actual size of the price movements. It's another strong sign that the model is struggling.

#### `MAPE`: 351,298%

> **Note:** This number is huge and looks scary, but you should **completely ignore it**. This metric breaks when the actual price change is very close to zero. Imagine the price barely changed (e.g., up 0.001%). Even a small prediction error (like 0.5%) will look like a gigantic percentage mistake. `SMAPE` is the reliable metric to use here.

**TO DOUBLE CHECK**

- review @validate_data_building.py
