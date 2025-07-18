# Logistic Regression for Marketing: Predicting Conversions by Source

[![R](https://img.shields.io/badge/R-4.3.1-blue?logo=r)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made With Love](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)](https://blog.marketingdatascience.ai)
[![CRAN](https://img.shields.io/badge/powered%20by-CRAN-blue)](https://cran.r-project.org/)

Created by [Joe Domaleski](https://blog.marketingdatascience.ai), July 2025.

This project demonstrates how to use logistic regression in a marketing context to predict website conversions based on traffic source. It uses synthetic data and includes complete R code and visualizations to help marketers understand and explain model results.

## 🧠 What This Project Covers
- Building and interpreting a logistic regression model
- Understanding log-odds and predicted probabilities
- Visualizing model output in a marketing-friendly way
- Exploring how traffic source affects conversion likelihood

## 📂 Files Included
- `simulated_sessions_with_timestamp.csv` — Synthetic dataset of marketing sessions with traffic source and conversion outcome.
- `logistic_regression_source_analysis.R` — Main R script that:
  - Prepares the data
  - Fits a logistic regression model
  - Generates plots for interpretation
- `README.md` — This file.

## 📊 Visualizations Included
- **Conversion Rate by Source** — Stacked bar chart showing actual outcomes
- **Mean Predicted Probability by Source** — Bar chart with confidence intervals and labels
- **Log-Odds Coefficient Plot** — Error bars vs. baseline category (Ad)
- **Predicted Probability Distribution** — Density plot by actual conversion

## 📖 Blog Post
This repo supports the blog post:  
📝 *What Happens Next? Predicting Marketing Outcomes with Logistic Regression*  
🔗 *(update with URL once published)*

## 📜 License
MIT License. See LICENSE file for details.
