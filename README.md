# 🏭 HVAC Energy Forecast & Repair Prediction

A machine learning + Streamlit app designed to predict commercial HVAC energy consumption and anticipate potential maintenance needs. Built for batch inference on time-series data with real-world deployment in mind.

## 📌 Overview

This project uses a finely-tuned **LightGBM regression model** trained on timestamped energy data to:

- 🔍 Forecast future HVAC energy usage
- ⚠️ Detect anomalies & generate **repair alerts**
- 📦 Serve results in a user-friendly **Streamlit dashboard**

It is designed for **real estate operators** and **facility managers** to better anticipate HVAC unit issues based on energy patterns.

---

## 🔧 Features

- ⏳ Time-aware feature engineering (lag features, cyclical encodings)
- ⚡ Fast, interpretable LightGBM predictions
- 📈 Residual-based repair signal detection
- 📁 Batch prediction via CSV upload
- 🖼️ Visual insights: prediction scatter, residual histograms
- 🚀 Deployable via Streamlit

---

