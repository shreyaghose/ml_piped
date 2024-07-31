# Understanding Data Splitting Errors in Data Pipelines
When constructing data pipelines for machine learning models, it's crucial to handle the splitting of data properly. Splitting the data at the wrong stage in the pipeline can lead to data leakage and invalid model evaluations. In this guide, we will explain the differences between two example pipelines and provide practical insights into the data splitting error.

## The Problem: Data Splitting Errors
The data splitting error occurs when the data is split into training and test sets at an incorrect stage of the pipeline. This typically happens when the data is split after some preprocessing steps have already been applied to the entire dataset, rather than splitting the raw data first. This can lead to data leakage, where information from the test set inadvertently influences the training process, resulting in overly optimistic performance estimates.

## Practical Advice
1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data.

2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.

3. **Pipeline Construction**: The correct order of operations - data extraction, data splitting, and then performing data transformation, model training, and evaluation, should be ensured in a pipeline.

## References
1. Gorman, K. and Bedrick, S., 2019, July. We need to talk about standard splits. In Proceedings of the conference. Association for Computational Linguistics. Meeting (Vol. 2019, p. 2786). NIH Public Access.
2. Oner, M.U., Cheng, Y.C., Lee, H.K. and Sung, W.K., 2020. Training machine learning models on patient level data segregation is crucial in practical clinical applications. medRxiv, pp.2020-04.
3. Lyu, Y., Li, H., Sayagh, M., Jiang, Z.M. and Hassan, A.E., 2021. An empirical study of the impact of data splitting decisions on the performance of AIOps solutions. ACM Transactions on Software Engineering and Methodology (TOSEM), 30(4), pp.1-38.
4. Tan, J., Yang, J., Wu, S., Chen, G. and Zhao, J., 2021. A critical look at the current train/test split in machine learning. arXiv preprint arXiv:2106.04525.