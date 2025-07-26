# pantheonplus-regression

This project applies simple machine learning models - such as **Linear Regression**, **Polynomial Regression**, **Decision Trees** and **Random Forests** to predict the **apparent magnitude($m_B$)** of Type Ia supernovae from the observed parameters:

- Redshift ($z$)
- Stretch ($x_1$)
- Color ($c$)

This project uses publicly available supernova datasets - **Pantheon+** and **DES 5-YEar** supernova compilations. The instructions to download these datasets are given in **data** folder. Please cite the respective papers if using these datasets in a work.

## Motivation

Type Ia supernova are standardizable candles used to measure cosmic distances. Here, we train regression models to predict the corrected/standardized apparent magnitudes of Type Ia supernovae from the redshift, stretch and color of the supernovae.

## Instructions

Run project_stage.py to download the data and train the models for use.

Then you can run app.py and access 127.0.0.1:5000 for a simple website that can predict type Ia supernova magnitudes from input.

## Project Structure

```text
pantheonplus-regression/
├── data/
│   └── README.md
├── notebooks/
│   ├── 0_pantheonplus_visualization.ipynb
│   ├── 1_model_baseline.ipynb
│   └── 2_model_advanced.ipynb
├── outputs/
│   └──models/
├── src/
│   ├── components
│   │   ├── handling
│   │   │   ├── exceptions.py
│   │   │   ├── logger.py
│   │   │   └── utils.py
│   │   └── model
│   │       ├── data_transformation.py
│   │       └── model_trainer.py
│   ├── notebooksrc
│   │   ├── preprocessing.py
│   │   └── regressors.py
│   └── pipeline
│       └── predict_pipeline.py
├── templates/
│   ├── home.html
│   └── index.html
└── README.md
```
