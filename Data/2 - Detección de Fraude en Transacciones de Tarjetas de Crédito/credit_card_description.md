# Dataset Description: Credit Card Fraud Detection

## Overview

The Credit Card Fraud Detection dataset is crucial for credit card companies to identify fraudulent transactions, ensuring customers are not charged for unauthorized purchases. The dataset comprises transactions made by credit cards in September 2013 by European cardholders.

## Data Source

The dataset is sourced from a research collaboration between Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

## Dataset Information

- **File Names**: credit_card_train.csv, credit_card_valid.csv, credit_card_test.csv
- **Purpose**: Dataset for fraud detection analysis
- **Format**: Comma-separated values (CSV)
- **Variables**:
  - `Log Amount`: Transaction amount.
  - `V1`, `V2`, ..., `V28`: Principal components obtained with PCA transformation.
  - `Class`: Target variable; takes value 1 in case of fraud and 0 otherwise.

## Context

The dataset contains transactions that occurred in two days, consisting of 492 frauds out of 284,807 transactions, making the dataset highly unbalanced. The positive class (frauds) accounts for 0.172% of all transactions.

It includes only numerical input variables resulting from a PCA transformation. Due to confidentiality reasons, the original features and further background information about the data cannot be provided. Features `V1` through `V28` are the principal components obtained with PCA, while `Log Amount` is the only feature not transformed with PCA.

## License

The dataset has been released for public use. Please cite the following works if you use this dataset:

- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. "Calibrating Probability with Undersampling for Unbalanced Classification." In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
- Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. "Learned lessons in credit card fraud detection from a practitioner perspective," Expert systems with applications, 41, 10, 4915-4928, 2014, Pergamon.
- Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. "Credit card fraud detection: a realistic modeling and a novel learning strategy," IEEE transactions on neural networks and learning systems, 29, 8, 3784-3797, 2018, IEEE.
- Dal Pozzolo, Andrea. "Adaptive Machine learning for credit card fraud detection." ULB MLG PhD thesis (supervised by G. Bontempi).
- Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. "Scarff: a scalable framework for streaming credit card fraud detection with Spark," Information fusion, 41, 182-194, 2018, Elsevier.
- Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. "Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization," International Journal of Data Science and Analytics, 5, 4, 285-300, 2018, Springer International Publishing.
- Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi. "Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection," INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019.
- Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi. "Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection," Information Sciences, 2019.
- Yann-Aël Le Borgne, Gianluca Bontempi. "Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook."
- Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi. "Incremental learning strategies for credit cards fraud detection," International Journal of Data Science and Analytics.