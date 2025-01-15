README for Assignment 2: Text Classification Using XGBoost

Overview

This project involves the development of a text classification model to classify text data into
different categories using the XGBoost algorithm. The notebook demonstrates data
preprocessing, feature extraction, model training, evaluation, and visualization techniques.
The aim is to achieve high accuracy by employing robust machine learning methods.

Setup
• Python version: 3.11.3
• IDE: Jupyter Notebook
• Libraries:
o pandas version: 2.2.3
o numpy version: 2.0.2
o nltk version: 3.9.1
o sklearn version: 1.5.2
o matplotlib version: 3.9.3
o seaborn version: 0.13.2
o wordcloud version: 1.9.4
o xgboost version: 2.1.3
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud xgboost.

General Libraries for Reading and Manipulation.
• pandas: For handling and manipulating data.
• numpy: For numerical operations.
For Text Preprocessing.
• sklearn.feature_extraction.text.CountVectorizer: For converting text to numerical
features.
• nltk.corpus.stopwords: For removing stopwords.
• nltk.stem.PorterStemmer: For stemming words.
• re: For text cleaning and regular expressions.
• nltk: For general text processing utilities.
Model Training and Evaluation.
• xgboost.XGBClassifier: The XGBoost classifier.
• sklearn.model_selection.train_test_split: For splitting the dataset into training and
testing sets.
• sklearn.model_selection.StratifiedKFold: For cross-validation.
• sklearn.metrics: For accuracy, precision, recall, F1-score, confusion matrix, and
precision-recall curves.
Data Visualization.
• matplotlib.pyplot: For creating static, interactive visualizations.
• seaborn: For enhanced data visualization.
Text Analysis.
• wordcloud.WordCloud: For generating word clouds.
• collections.Counter: For counting word frequencies.
Ignore Warnings.
• warnings: To suppress warnings for cleaner outputs.
