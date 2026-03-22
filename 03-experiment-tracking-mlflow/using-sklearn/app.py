

# import pandas as pd
# import pickle
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn import datasets


# import numpy as np
# from sklearn.metrics import accuracy_score

# # df=pd.read_csv('practical4/using-sklearn/Iris.csv')
# # df.drop(columns=['Id'],errors='ignore',inplace=True)

# # X=df.iloc[:,:-1].values
# # y=df.iloc[:,-1].values


# iris=datasets.load_iris()
# X = iris.data          
# y = iris.target    



# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# pipeline=Pipeline([
#     ('scaler',StandardScaler()),    
#     ('classifier',GaussianNB())])
# pipeline.named_steps['classifier'].priors=[0.3,0.4,0.3]
# pipeline.fit(X_train,y_train)
# y_pred=pipeline.predict(X_test)
# accuracy=accuracy_score(y_test,y_pred)

# print(f'Accuracy: {accuracy*100:.3f}%')

# with open('pipeline.pkl','wb') as f:
#     pickle.dump(pipeline,f)

# print("Pipeline saved as 'pipeline.pkl'")

# with open('pipeline.pkl','rb') as f:
#     loaded_pipeline=pickle.load(f)

# sample_data=np.array([[5.1,3.5,1.4,0.2]])
# predicted_class=loaded_pipeline.predict(sample_data)                                                                                    

# print(f'Predicted class for sample data {sample_data[0]}: {predicted_class[0]}')



import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Wine Dataset
# -----------------------------
wine = datasets.load_wine()
X = wine.data          # 13 numerical features
y = wine.target        # 3 classes (0, 1, 2)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Build Pipeline (Scaler + NB)
# -----------------------------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

# Optional: adjust priors (class weights)
pipeline.named_steps['classifier'].priors = [0.3, 0.4, 0.3]

# -----------------------------
# Train and Evaluate
# -----------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.3f}%')

# -----------------------------
# Save the Trained Pipeline
# -----------------------------
with open('wine_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Pipeline saved as 'wine_pipeline.pkl'")

# -----------------------------
# Load the Pipeline and Predict
# -----------------------------
with open('wine_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Example sample from Wine dataset (first record)
sample_data = np.array([X_test[0]])  # reshape(1, -1) not needed since X_test[0] is already 1D
predicted_class = loaded_pipeline.predict(sample_data)

print(f'Predicted class for sample data {sample_data[0]}: {predicted_class[0]}')
print(f'Actual class: {y_test[0]}')

# Optional: map numeric label to readable name
print(f'Predicted label name: {wine.target_names[predicted_class[0]]}')
