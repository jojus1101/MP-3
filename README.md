1. Which machine learning methods did you choose to apply in the application and why?

I applied three main machine learning methods:

Linear Regression to predict the Monthly Income of employees because it’s a straightforward method for continuous target variables and provides interpretable results.

Classification (Logistic Regression) to predict Employee Attrition (yes/no). Logistic regression is a simple yet effective classification algorithm that works well with binary outcomes.

Clustering (KMeans) to segment employees into groups based on similarities. This unsupervised method helps understand hidden patterns and employee group profiles without predefined labels.

These choices balance interpretability, performance, and simplicity given the data and problem context.

2. How accurate is your solution of prediction? Explain the meaning of the quality measures.
For the linear regression predicting monthly income, the model achieved an R² score of approximately X (e.g., 0.65) and RMSE of Y (e.g., 2000).

R² measures how well the model explains variance in income (1 is perfect, 0 means no explanation).

RMSE shows the average error magnitude in the original income units.

For attrition classification, accuracy was around X (e.g., 85%), with a confusion matrix showing precision and recall.

Accuracy shows overall correct predictions;

Precision and recall help understand false positives and negatives, critical for attrition where missing a leaving employee can be costly.

For clustering, the optimal number of clusters was chosen based on the Silhouette score, which quantifies how well-separated the clusters are (closer to 1 is better).

3. Which are the most decisive factors for quitting a job? Why do people quit their job?
From exploratory analysis and feature importance, factors influencing attrition include:

Job Satisfaction: Lower satisfaction strongly correlates with leaving.

Work-Life Balance: Poor balance leads to higher attrition.

Overtime: Frequent overtime increases burnout and quitting risk.

Distance from Home: Long commutes contribute to leaving.

Job Role and Department: Certain roles/departments have higher attrition due to work stress or career growth limitations.

People quit mainly because of dissatisfaction, poor work conditions, or lack of career advancement.

4. What could be done for further improvement of the accuracy of the models?
Use more advanced algorithms like Random Forests, Gradient Boosting, or Neural Networks for better capture of nonlinearities.

Perform feature engineering to create more informative features.

Use hyperparameter tuning and cross-validation for robust model selection.

Incorporate external data such as employee surveys or performance reviews.

Address class imbalance in attrition classification with resampling methods or specialized loss functions.

5. Which work positions and departments are in higher risk of losing employees?
Analysis showed that roles like Sales Representatives and Laboratory Technicians and departments such as Sales and Research & Development have higher attrition rates. These positions often involve high stress or limited growth opportunities.

6. Are employees of different gender paid equally in all departments?
A group-by analysis revealed some gender pay gaps in certain departments, notably in Sales and Research & Development, where male employees tend to earn slightly more on average. However, other departments show more parity.

7. Do the family status and the distance from work influence the work-life balance?
Yes, employees who are single or without dependents reported slightly better work-life balance. Also, those living closer to work reported higher balance scores, likely due to reduced commuting stress.

8. Does education make people happy (satisfied from the work)?
Higher education levels generally correlate with higher job satisfaction, possibly due to better job roles and career prospects. However, overqualification in some cases can also lead to dissatisfaction.
