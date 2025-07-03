# project-by-Medical-Fraud-Detection-in-healthcare-using-data-balancing-and-Machine-learning-Technique
Medical fraud has become a major concern in the healthcare industry, leading to substantial financial losses and misuse of medical resources. This project aims to analyze and detect fraudulent claims in healthcare systems by leveraging data balancing techniques and machine learning (ML) algorithms.

The project begins with the collection of a healthcare claims dataset, which typically includes features such as patient ID, provider details, claim amount, diagnosis codes, and claim status (fraud or not fraud). One of the key challenges in medical fraud detection is the class imbalance—fraudulent cases are significantly fewer compared to genuine ones. This imbalance can mislead machine learning models into favoring the majority class, resulting in poor fraud detection accuracy.

To address this, data balancing techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are used. SMOTE generates synthetic examples of the minority class (fraudulent claims) to balance the dataset, improving model sensitivity to fraud patterns.

After preprocessing, several ML algorithms—such as Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM)—are trained on the balanced data. These models learn patterns and anomalies associated with fraudulent behavior in healthcare claims.

The performance of each model is evaluated using metrics like accuracy, precision, recall, and F1-score. Among these, recall is especially important in fraud detection, as it measures how effectively the model identifies actual frauds.
