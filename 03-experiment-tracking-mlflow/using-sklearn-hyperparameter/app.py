# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score
# import numpy as np
# from sklearn import datasets
# # -----------------------------
# # Load dataset
# # -----------------------------

# # df=pd.read_csv('practical4/using-sklearn/Iris.csv')
# # df.drop(columns=['Id'],errors='ignore',inplace=True)

# # X=df.iloc[:,:-1].values
# # y=df.iloc[:,-1].values


# iris=datasets.load_iris()
# X = iris.data          
# y = iris.target    



# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -----------------------------
# # Pipeline – sklearn with Hyper-Parameter Tuning
# # -----------------------------
# # Pipeline: scaling + classifier
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('clf', GaussianNB())
# ])

# # Define hyperparameter grid
# param_grid = {
#     'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # GaussianNB smoothing parameter
# }

# # Grid search with cross-validation
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # -----------------------------
# # Best parameters and accuracy
# # -----------------------------
# print("Best Parameters:", grid_search.best_params_)

# y_pred = grid_search.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Tuned Pipeline Accuracy: {accuracy:.2f}")

# # Save the tuned pipeline
# with open("pipeline_tuned.pkl", "wb") as f:
#     pickle.dump(grid_search.best_estimator_, f)

# print("Tuned pipeline saved as 'pipeline_tuned.pkl'")

# # -----------------------------
# # Load the saved pipeline
# # -----------------------------
# with open("pipeline_tuned.pkl", "rb") as f:
#     pipeline_tuned = pickle.load(f)

# # Example input for prediction (same number of features as training data)
# X_new = np.array([[5.1, 3.5, 1.4, 0.2]])

# # Make prediction
# prediction = pipeline_tuned.predict(X_new)

# # Print result
# print(f"Predicted class: {prediction[0]}")



import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets

# -----------------------------
# Load Wine dataset
# -----------------------------
wine = datasets.load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Pipeline – scaling + classifier
# -----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GaussianNB())
])

# Define hyperparameter grid
param_grid = {
    'clf__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# -----------------------------
# Best parameters and accuracy
# -----------------------------
print("Best Parameters:", grid_search.best_params_)

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tuned Pipeline Accuracy: {accuracy:.2f}")

# -----------------------------
# Save the tuned pipeline
# -----------------------------
with open("wine_pipeline_tuned.pkl", "wb") as f:
    pickle.dump({
        'model': grid_search.best_estimator_,
        'class_names': class_names,
        'scaler': grid_search.best_estimator_.named_steps['scaler']
    }, f)

print("Tuned pipeline saved as 'wine_pipeline_tuned.pkl'")

# -----------------------------
# Load the saved pipeline
# -----------------------------
with open("wine_pipeline_tuned.pkl", "rb") as f:
    model_data = pickle.load(f)
    pipeline_tuned = model_data['model']
    class_names = model_data['class_names']

# Example input (5 features subset from your Flask example)
# The wine dataset actually has 13 features, so you must give all 13 values!
# Below is one valid sample from the dataset:
X_new = np.array([[13.2, 2.77, 2.51, 18.5, 98, 1.82, 0.47, 0.86, 5.4, 1.1, 3.3, 820, 1.04]])

# Make prediction
prediction = pipeline_tuned.predict(X_new)

print(f"Predicted class index: {prediction[0]}")
print(f"Predicted wine class name: {class_names[prediction[0]]}")
