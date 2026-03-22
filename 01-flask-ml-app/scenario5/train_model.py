import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

# Load Wine dataset and use only 5 key features
wine = datasets.load_wine()
# Select 5 most important features: alcohol, flavanoids, color_intensity, hue, proline
feature_indices = [0, 6, 9, 10, 12]  # alcohol, flavanoids, color_intensity, hue, proline
X = wine.data[:, feature_indices]
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy*100:.3f}%')

# Save model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'class_names': wine.target_names,
    'feature_names': ['Alcohol', 'Flavanoids', 'Color Intensity', 'Hue', 'Proline']
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved as 'model.pkl'")
