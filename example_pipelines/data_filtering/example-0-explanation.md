# Understanding Data Filtering Errors in Data Pipelines
When constructing data pipelines for machine learning models, it is crucial to handle data filtering appropriately. Data filtering errors occur when the selection or exclusion of data points during preprocessing alters the distribution of feature classes, leading to biased model evaluations. In this guide, we will explain the impact of data filtering issues, how they occur, and practical steps to avoid them.

## The Problem: Data Filtering Errors
Data filtering errors happen when data is filtered in a way that changes the proportion of feature classes, such as age, race, or gender. This can inadvertently introduce bias into the model, especially if the filtered feature class is a protected group. Applying transformations like selection and joins arbitrarily can harm the fairness of the model, leading to skewed performance metrics and unfair treatment of certain groups.

## Practical Advice

1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data. This ensures that any filtering applied afterward does not introduce bias into the model evaluation process.

2. **Proportional Filtering**: Filtering should be done carefully to maintain the original distribution of feature classes. This can involve stratified sampling or other techniques to ensure the filtered dataset still represents the population accurately.

2. **Separate Preprocessing**: Apply preprocessing steps separately to the training and test sets to avoid changing the distribution of feature classes in the test set ensures that the evaluation metrics reflect the model's performance on a representative sample of the data.

3. **Pipeline Construction**: The distribution of feature classes should be monitored before and after filtering, validating that the filtered data maintains the original proportions and does not introduce unintended bias.

## References
1. Stoyanovich, J., Howe, B. and Jagadish, H.V., 2020. Responsible data management. Proceedings of the VLDB Endowment, 13(12).
2. García, V., Marqués, A.I. and Sánchez, J.S., 2012. On the use of data filtering techniques for credit risk prediction with instance-based models. Expert Systems with Applications, 39(18), pp.13267-13276.
3. Khalid, F., Hanif, M.A., Rehman, S., Qadir, J. and Shafique, M., 2019, March. Fademl: Understanding the impact of pre-processing noise filtering on adversarial machine learning. In 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE) (pp. 902-907). IEEE.
4. García-Gil, D., Luengo, J., García, S. and Herrera, F., 2019. Enabling smart data: noise filtering in big data classification. Information Sciences, 479, pp.135-152. 