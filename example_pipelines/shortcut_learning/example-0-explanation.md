# Understanding Shortcut Learning in Data Pipelines
Shortcut learning arises when systems use unintended learning due to superficial correlations in the data. Geirhos et al. define shortcuts as a set of developer decision rules that perform well on independent test data that are identically distributed (i.i.d) to the training set but fail on out-of-distribution (o.o.d) test sets that are different from both i.i.d and training sets. Shortcuts are not obvious and cannot be detected by the human eye.

## The Problem: Shortcut Learning
Shortcut learning occurs when models exploit spurious correlations in the training data, leading to poor generalization on o.o.d test sets. This issue can arise due to:

1. Covariate Shift: Changes in the distribution of input data.
2. Unintended Anti-Causal Learning: Learning unintended causal relationships.
3. Clever Hans Effect: Models learning to rely on irrelevant features.
4. Dataset Biases: Inherent biases in the dataset that lead to spurious correlations.

## Practical Advice
1. **Feature Importance Evaluation**: Regularly assessing the importance of features ensures that the model is not relying on spurious correlations.

2. **Data Augmentation**: Data augmentation techniques introduce variability and reduce reliance on specific features.

3. **Domain Adaptation**: Application of domain adaptation methods to ensure the model generalizes well to o.o.d test sets.

4. **Cross-Validation**: Usage of robust cross-validation techniques could help with shortcut learning detection.

5. **Regular Model Audits**: Conducting regular audits of the model's performance on diverse datasets help with identification and mitigation of shortcut learning.

## References
1. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. and Wichmann, F.A., 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11), pp.665-673.
2. Wang, M. and Deng, W., 2018. Deep visual domain adaptation: A survey. Neurocomputing, 312, pp.135-153.
3. Banerjee, I., Bhattacharjee, K., Burns, J.L., Trivedi, H., Purkayastha, S., Seyyed-Kalantari, L., Patel, B.N., Shiradkar, R. and Gichoya, J., 2023. “Shortcuts” causing bias in radiology artificial intelligence: causes, evaluation and mitigation. Journal of the American College of Radiology.
4. Dodge, S. and Karam, L., 2019. Human and DNN classification performance on images with quality distortions: A comparative study. ACM Transactions on Applied Perception (TAP), 16(2), pp.1-17.
5. Alcorn, M.A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.S. and Nguyen, A., 2019. Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4845-4854).