# Deep Learning for Tropical Cyclone Genesis Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![NASA MERRA-2](https://img.shields.io/badge/Data-NASA%20MERRA--2-blue)](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)

## Overview

This project investigates the application of deep learning to forecast **Tropical Cyclone Genesis**—the initial formation of a hurricane—within a 24-hour window.

By leveraging meteorological data from **NASA’s MERRA-2** system, we model genesis prediction as a **semantic segmentation task**. The system processes 2D atmospheric fields (satellite and feature imagery) to output spatial probability maps, identifying specific regions where a cyclone is likely to form. This approach aims to improve early warning systems for coastal resilience.

## Key Features

* **Rare Event Detection:** Designed to handle the extreme class imbalance of genesis events (which are rare) versus non-genesis ocean areas.
* **Deep Learning Architectures:** Benchmarks three state-of-the-art segmentation models: **U-Net**, **ResSegNet**, and **Attention U-Net**.
* **Multi-Variable Input:** Ingests a 22-channel tensor consisting of kinematic and thermodynamic variables (e.g., Sea Level Pressure, Vorticity, Humidity) across multiple pressure levels.
* **Robust Evaluation:** Validated against historical tropical cyclone data from both the **Atlantic** and **Pacific** basins.

## System Architecture

The pipeline is structured into three phases:

1.  **Data Processing:**
    * **Source:** NASA MERRA-2 Reanalysis.
    * **Preprocessing:** Normalization (Z-score) of 22 atmospheric variables.
    * **Labeling:** Generation of binary segmentation masks based on historical storm tracks (Positive label = Genesis within 24h).

2.  **Model Framework:**
    * **U-Net:** Standard encoder-decoder baseline.
    * **ResSegNet:** Incorporates residual blocks for deeper feature extraction.
    * **Attention U-Net (Best Model):** Utilizes **Attention Gates** to filter background noise and focus on relevant spatial features of developing storms.

3.  **Training Strategy:**
    * **Loss Function:** **Focal Loss** is used to prioritize "hard" positive samples and mitigate the impact of class imbalance.
    * **Optimization:** AdamW optimizer with Early Stopping to prevent overfitting.

## 📊 Key Results

* **Best Performance:** The **Attention U-Net** achieved the highest accuracy and reliability, with a **ROC-AUC of 0.72**.
* **Architecture Comparison:**
    * *Attention U-Net* successfully localized genesis regions for major test cases (e.g., Hurricane Haishen, Hurricane Laura).
    * *ResSegNet* underperformed (AUC 0.43), failing to capture the complex spatial dependencies.
    * *Standard U-Net* provided reasonable detection but suffered from higher false-positive rates.
* **Conclusion:** The study confirms that attention mechanisms are critical for suppressing noise in meteorological data, making them superior for rare-event forecasting.

## License

This project is for academic research purposes.
