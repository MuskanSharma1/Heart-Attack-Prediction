# Heart-Attack-Prediction

# Description:
Using a combination of robust Python modules such as NumPy, Pandas, Matplotlib, Seaborn, and Plotly Express, we have created a model to predict cardiac attacks. The goal of this model is to analyze a dataset that includes a variety of heart attack symptoms and risk factors, including age, sex, type of chest pain (cp), resting blood pressure (trestbps), cholesterol level (chol), fasting blood sugar (fbs), maximum heart rate achieved (thalach), exercise-induced angina (exang), ST depression induced by exercise relative to rest (oldpeak), slope of the peak exercise ST segment (slope), number of major vessels colored by fluoroscopy (ca), thalassemia (thal), and the target variable that indicates whether or not a heart attack is occurring.


To build this model, I've utilized the following steps:

1. Data Preprocessing: To address missing values, encode categorical variables, and guarantee data quality, we cleaned and preprocessed the dataset using Pandas.

2. Exploratory Data Analysis (EDA): Using Matplotlib and Seaborn, we investigated correlations between variables and showed the distribution of each feature. 

3. Feature Engineering: To increase our model's capacity for prediction, we carefully chosen pertinent characteristics and, if needed, created additional ones.

4. Machine Learning Model Selection: To determine the most accurate model for heart attack prediction, we explored with a variety of machine learning algorithms, including logistic regression, decision trees, random forests, and gradient boosting.

5. Model assessment: We used relevant assessment criteria, including accuracy, precision, recall, and F1-score, to assess each model's performance. Additionally, we used ROC curves and confusion matrices to visualize the model's performance.

6. Deployment: In order to determine the likelihood of a heart attack given a collection of symptoms, we lastly installed the top-performing model. Healthcare practitioners can use this model to determine a patient's risk of having a heart attack and to take appropriate preventive action.


Description of Algorithms Used:

1. Logistic Regression:-
Logistic regression is a supervised learning algorithm used for binary classification tasks, such as predicting whether an event will occur or not (e.g., heart attack). Despite its name, logistic regression is a linear model that uses the logistic function (also known as the sigmoid function) to model the probability of the target variable belonging to a particular class. It works by fitting a logistic curve to the input features, which transforms the output into a probability between 0 and 1. In our heart attack prediction model, logistic regression is used to estimate the probability of a patient experiencing a heart attack based on their symptoms and risk factors. The model learns the coefficients of the input features and applies them to calculate the probability of a positive outcome (heart attack). Logistic regression is computationally efficient, interpretable, and well-suited for binary classification tasks with linear decision boundaries.

2. Random Forest Classifier:-
Random forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting. Each decision tree in the random forest is trained on a random subset of the training data and a random subset of the input features. During prediction, the random forest aggregates the predictions of all individual trees to make a final prediction. Random forests are highly flexible and robust, capable of capturing complex nonlinear relationships in the data. They are particularly effective for classification tasks, including binary classification tasks like heart attack prediction. In our heart attack prediction model, the random forest classifier leverages the diversity of decision trees to accurately classify patients into heart attack and non-heart attack groups based on their symptoms and risk factors. Random forests are known for their high accuracy, scalability, and resistance to overfitting, making them a popular choice for various machine learning applications.




