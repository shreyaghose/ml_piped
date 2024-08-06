# PIPED - Practical Issues and Pipeline Examples Dataset

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Welcome to PIPED, a collection of example pipelines demonstrating various data preparation issues and their solutions. This repository is designed to help data practitioners identify common pitfalls in data processing and learn how to address them effectively.

## Overview
Data preparation is a critical step in any data-driven project. Mistakes in this phase can lead to misleading results, faulty analyses, and ultimately, incorrect decisions. This repository provides practical examples of 23 common data preparation issues, along with flawed and corrected pipeline implementations for each issue.

## Structure
Each issue is organized in a dedicated folder containing:

1. **Flawed Pipeline**: A demonstration of how the issue can manifest in a data pipeline.
2. **Corrected Pipeline**: A solution or best practice that mitigates the identified issue.
3. **Documentation**: A .md file explaining the issue, potential consequences, and practical advice on avoiding or fixing it.

## Datasets
The pipelines in this repository utilize a variety of datasets to illustrate different data preparation issues. Below is an overview of the datasets used:

1. **COMPAS Dataset**:
Description: A dataset containing information on criminal defendants, used to predict recidivism risk.
Use Case: Demonstrates issues like data leakage, bias, and ethical considerations in predictive modeling.

2. **Adult Income**:
Description: A dataset containing demographic information used to predict whether an individual's income exceeds $50,000 per year.
Use Case: Illustrates data preprocessing steps such as handling missing values, encoding categorical variables, and scaling features. It also addresses class imbalance and the ethical implications of predictive modeling on socioeconomic data.

3. **Titanic Dataset**:
Description: Historical data from the Titanic voyage, including details about passengers and their survival status.
Use Case: Illustrates challenges in data imputation, handling missing values, and dealing with categorical variables.

4. **Diabetes Indicator Dataset**:
Description: A dataset used for predicting diabetes outcomes based on various health indicators.
Use Case: Highlights issues such as outlier management, normalization, and the handling of imbalanced data.

5. **Impact of Alcohol on University Grades**:
Description: Data on university students' alcohol consumption and its impact on their academic performance.
Use Case: Explores issues related to correlation vs. causation, confounding variables, and data aggregation errors.


## Issues Covered
The repository includes examples of the following issues, among others:
Aggregation Errors
Data Imputation Challenges
Data Leakage
Inconsistent Data Types
Missing Data Handling
Outlier Management


## How to Use This Repository
Explore the Issues: Browse through the issues to understand common data preparation problems.
Review the Examples: Examine the flawed and corrected pipelines to see how these issues can occur and be resolved.
Read the Documentation: Each issue includes a markdown file with detailed explanations and advice.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE] (LICENSE.txt) file for details.