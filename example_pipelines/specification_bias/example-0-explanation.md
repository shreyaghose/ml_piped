# Understanding Specification Bias in Data Pipelines
Specification Bias refers to errors or biases introduced when selecting which features to include in a learning task, either intentionally or unintentionally. This type of bias can arise when protected attributes are omitted but their proxies remain, leading to unintended biases in the model.

## The Problem: Specification Bias
With multiple protected attributes, specification bias can occur due to:

1. **Omission of Multiple Protected Attributes**: Removing multiple protected attributes from the feature set, while their proxies may still influence the model.

2. **Proxy Attributes**: Features that correlate with any of the protected attributes can still impart bias if they are not removed.

3. **Irrelevant Features**: Features unrelated to the target but correlated with protected attributes may introduce bias.

## Practical Advice
1. **Identify Proxies for Each Protected Attribute**: Correlation analysis should be used to detect features that proxy for any of the protected attributes and these features should be removed from the dataset.

2. **Comprehensive Feature Selection**: All features that could act as proxies for protected attributes should be accounted for and excluded.

3. **Transparent Documentation**: The rationale behind the selection and exclusion of features should be documented to maintain transparency.

4. **Bias Audits**: The dataset should be regularly reviewed to identify and address any new proxy attributes that may emerge.

## References
1. Hellström, T., Dignum, V. and Bensch, S., 2020. Bias in Machine Learning--What is it Good for?. arXiv preprint arXiv:2004.00686.