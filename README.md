# pantheonplus-regression

This project applies simple machine learning models - such as **Linear Regression**, **Polynomial Regression**, **Decision Trees** and **Random Forests** to predict the **apparent magnitude(`m_B$`)** of Type Ia supernovae from the observed parameters:

- Redshift (`$z$`)
- Stretch (`$x_1$`)
- Color (`$c$`)

This project uses publicly available supernova datasets - **Pantheon+** and **DES 5-YEar** supernova compilations.

## Motivation

Type Ia supernova are standardizable candles used to measure cosmic distances. Here, we train regression models to predict the corrected/standardized apparent magnitudes of Type Ia supernovae from the redshift, stretch and color of the supernovae.

## Project Structure

```text
pantheonplus-regression/
├── data/
│   ├── download_files.py
│   └── README.md
├── notebooks/
│   ├── 0_pantheonplus_visualization.ipynb
│   ├── 1_model_baseline.ipynb
│   └── 2_model_advanced.ipynb
├── src/
│   ├── preprocessing.py
│   └── regressors.py
└── README.md
```
