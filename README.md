# ML PIPED - Practical Issues in Pipelines -  Example Dataset

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Welcome to ML PIPED, a collection of example pipelines demonstrating various data preparation issues and their solutions. This repository is designed to help data practitioners identify common pitfalls in data processing and learn how to address them effectively.

## Overview
Data preparation is a critical step in any data-driven project. Mistakes in this phase can lead to misleading results, faulty analyses, and ultimately, incorrect decisions. This repository provides practical examples of 21 common data preparation issues, along with flawed and corrected pipeline implementations for each issue.

## Structure
Each issue is organized in a dedicated folder containing:

1. **Flawed Pipeline**: A demonstration of how the issue can manifest in a data pipeline.
2. **Corrected Pipeline**: A solution or best practice that mitigates the identified issue.
3. **Documentation**: A .md file explaining the issue, potential consequences, and practical advice on avoiding or fixing it.

## Datasets
The pipelines in this repository utilize a variety of datasets to illustrate different data preparation issues. Below is an overview of the datasets used:

1. **COMPAS Dataset**:<br>
Description: A dataset containing information on criminal defendants, used to predict recidivism risk. <br>
Use Case: Demonstrates issues like data leakage, bias, and ethical considerations in predictive modeling.

2. **Adult Income**:<br>
Description: A dataset containing demographic information used to predict whether an individual's income exceeds $50,000 per year.<br>
Use Case: Illustrates data preprocessing steps such as handling missing values, encoding categorical variables, and scaling features. It also addresses class imbalance and the ethical implications of predictive modeling on socioeconomic data.

3. **Titanic Dataset**:<br>
Description: Historical data from the Titanic voyage, including details about passengers and their survival status.<br>
Use Case: Illustrates challenges in data imputation, handling missing values, and dealing with categorical variables.

4. **Diabetes Indicator Dataset**:<br>
Description: A dataset used for predicting diabetes outcomes based on various health indicators.<br>
Use Case: Highlights issues such as outlier management, normalization, and the handling of imbalanced data.

5. **Impact of Alcohol on University Grades**:<br>
Description: Data on university students' alcohol consumption and its impact on their academic performance.<br>
Use Case: Explores issues related to correlation vs. causation, confounding variables, and data aggregation errors.


## Issues Covered
The repository includes examples of the issues that can be found during the data preparation stage in a machine learning pipeline. The selection of the issues detailed in the provided categorization framework is a result of a comprehensive literature review encompassing over 35 identified problems and biases, in different stages of the pipeline. Each issue was chosen based on its prevalence in the research, its influence on fairness, interpretability, and reproducibility, and its relevance across different pipeline phases, from input datasets through to model training and evaluation. Some example issues are:
1. [Data Leakage](example_pipelines\data_leakage)
2. [Shortcut Learning](example_pipelines\shortcut_learning)
3. [Anonymization Errors](example_pipelines\data_anonymization)


## How to Use This Repository
1. Exploring the Issues: Browsing through the issues will help understand common data preparation problems.
2. Reviewing the Examples: Examining the flawed and corrected pipelines provides insights on how these issues can occur and be resolved.
3. Reading the Documentation: Each issue includes a markdown file with detailed explanations and advice.
4. Repeating Experimentation: The experimentation can be tailored and fit to any dataset and repeated.

## License
This project is licensed under the Apache License 2.0. Refer to the [LICENSE](LICENSE.txt) file for details.