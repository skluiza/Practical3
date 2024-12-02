# Practical3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and preprocess data
data = pd.read_csv("Phishing_Legitimate_full.csv")
data.rename(columns={'CLASS_LABEL': 'labels'}, inplace=True)
data = data.astype({col: 'float32' if data[col].dtype == 'float64' else 'int32' for col in data.columns})

# Plot label distribution
data['labels'].value_counts().plot(kind='bar')
plt.show()


# Mutual information feature selection
X, y = data.drop(['id', 'labels'], axis=1), data['labels']
mi_scores = pd.Series(mutual_info_classif(X, y), index=X.columns).sort_values(ascending=False)


# Plot MI Scores
mi_scores.plot(kind='barh', title="Top Features", figsize=(20, 10))
plt.show()

# Train and evaluate Random Forest with top features
def evaluate_model(top_n):
    features = mi_scores.head(top_n).index
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, shuffle=True)
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

  evaluate_model(32)
