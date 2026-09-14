# Traffic Volume Prediction with Linear and Poisson Models

This project investigates how linear and generalized linear models can predict hourly traffic volume from temporal and weather-related features. It uses the **Metro Interstate Traffic Volume** dataset and compares models not only by predictive performance, but also by their statistical assumptions, residual behavior, and ability to generalize to future observations.

The work was developed for **MC886 — Introduction to Machine Learning, Assignment 1**.

## Project goals

- Explore the dataset and identify missing intervals, outliers, redundant features, and temporal patterns.
- Build a leakage-safe preprocessing pipeline using only statistics learned from the training period.
- Implement linear and Poisson regression with gradient descent.
- Evaluate how polynomial terms and feature interactions affect model capacity and overfitting.
- Compare Gaussian and Poisson assumptions for a count-valued target.
- Analyze predictions and residuals using numerical metrics and diagnostic plots.

## Dataset

The data records westbound traffic on Interstate 94 between Minneapolis and St. Paul, Minnesota. Each row combines an hourly vehicle count with calendar and weather information.

The notebook uses **26,677 observations** from 2015 to 2018 and includes variables such as:

- hour, weekday, month, season, year, holiday, and working-day indicators;
- temperature, rainfall, snowfall, and cloud coverage;
- general and detailed weather descriptions;
- hourly traffic volume, used as the prediction target.

Source: [Metro Interstate Traffic Volume — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> The notebook expects a compatible file named `traffic_volume.csv` in the same directory. The course-provided version should be used to reproduce the reported results.

## Methodology

### Temporal split

The observations are kept in chronological order. The first 80% form the training set, and the final 20% form the test set. The test period remains untouched until the final evaluation, preventing temporal leakage.

### Exploratory analysis and preprocessing

The pipeline applies the following decisions:

- removes `instant`, which is only an identifier;
- uses `date` for chronological ordering and splitting, but not as a direct predictor;
- removes the physically implausible training value `rain_1h = 9831.3`;
- drops `weather_desc` and retains the more compact `weather` feature;
- combines `weekday`, `holiday`, and `workingday` into `day_type`;
- uses `mnth` as the default seasonal representation instead of simultaneously using `mnth` and `season`;
- standardizes numerical features using training-set statistics only;
- applies one-hot encoding with categories learned exclusively from the training set.

The final preprocessed feature matrix contains **53 predictors** before polynomial expansion.

### Models

Four configurations are compared:

1. **Linear regression** — additive linear predictor optimized with mean squared error.
2. **Degree-2 polynomial regression** — linear regression after adding quadratic terms and pairwise interactions.
3. **Poisson regression** — generalized linear model with a log link, producing strictly positive predictions.
4. **Degree-2 polynomial Poisson regression** — Poisson regression applied to the expanded feature space.

The regression classes, evaluation metrics, and polynomial expansion are implemented directly in the notebook. The models are trained with gradient descent rather than imported estimators.

## Results

| Model | Split | MSE | R² | MAE |
|---|---:|---:|---:|---:|
| Linear | Train | 612,923 | 0.841 | 579 |
| Linear | Test | 587,394 | 0.851 | 565 |
| Degree-2 polynomial | Train | 160,189 | 0.958 | 265 |
| Degree-2 polynomial | Test | 615,704 | 0.844 | 372 |
| Poisson | Train | 496,434 | 0.871 | 504 |
| Poisson | Test | 470,233 | 0.881 | 485 |
| Degree-2 polynomial Poisson | Train | 213,716 | 0.945 | 317 |
| **Degree-2 polynomial Poisson** | **Test** | **334,011** | **0.915** | **364** |

The ordinary degree-2 polynomial model achieves the lowest training error but loses most of that advantage on the test period, indicating overfitting. The degree-2 polynomial Poisson model provides the best test MSE and R² while preserving the nonnegative nature of traffic counts.

The results also show why multiple metrics matter: the Gaussian polynomial model has a much lower test MAE than the linear baseline, but worse MSE and R² because a small number of large errors are penalized heavily by squared error.

## Diagnostic analysis

The notebook includes:

- predicted-versus-observed plots;
- hourly traffic profiles for working and non-working days;
- short time-series comparisons of actual and predicted values;
- residual histograms and residuals over time;
- Q–Q plots for models based on the Gaussian assumption;
- residual-versus-fitted analysis for Poisson regression.

These diagnostics reveal interactions between hour and day type, systematic errors at very low traffic volumes, heavy residual tails, and the limitations of assuming constant Gaussian variance for count data.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn

Install the dependencies with:

```bash
python -m pip install jupyter numpy pandas matplotlib scipy scikit-learn
```

## Running the project

1. Clone or download the repository.
2. Place `traffic_volume.csv` in the same directory as the notebook.
3. Start Jupyter:

```bash
jupyter notebook
```

4. Open the project notebook.
5. Run all cells in order.

## Main conclusions

- Traffic volume has a strong nonlinear temporal structure, especially across hours and day types.
- A simple linear model is a stable and interpretable baseline, but it cannot fully represent important interactions.
- Polynomial expansion greatly increases capacity and can overfit when model selection relies only on training error.
- The Poisson log link is better aligned with a positive count target than unconstrained Gaussian regression.
- Among the evaluated configurations, the degree-2 polynomial Poisson model offers the strongest test performance.

## Authors

- Ana Carolina de Almeida Cardoso
- Francisco Vinicius Sousa Guedes
- Pedro Damasceno Vasconcellos

## Academic context

**Course:** MC886 — Introduction to Machine Learning  
**Professor:** Marcelo da Silva Reis  
**Teaching Assistant:** Heigon Alafaire Soldera Pires

