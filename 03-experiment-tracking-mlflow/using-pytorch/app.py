# 




import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
from itertools import product
import torch.serialization
from sklearn import datasets

# ----------------------------
# Load Wine Dataset
# ----------------------------
wine = datasets.load_wine()

# Create a DataFrame (optional for readability)
dataset_df = pd.DataFrame(wine.data, columns=wine.feature_names)
dataset_df['target'] = wine.target

# Feature Matrix and Target Vector
feature_data = dataset_df.iloc[:, :-1].values  # 13 features
target_labels_encoded = dataset_df.iloc[:, -1].values  # Already numeric (0,1,2)

# Split into Training and Testing Sets
X_train_full, X_validation_test, y_train_full, y_validation_test = train_test_split(
    feature_data, target_labels_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# Define Neural Network Pipeline
# ----------------------------
class NeuralNetPipeline:
    def __init__(self, neuron_count=16):
        self.feature_scaler = StandardScaler()
        # Input features: 13 (Wine dataset)
        # Output classes: 3 (Wine classes)
        self.network = nn.Sequential(
            nn.Linear(13, neuron_count),
            nn.ReLU(),
            nn.Linear(neuron_count, 3)
        )

    def preprocess_features(self, feature_matrix):
        matrix_copy = feature_matrix.copy()
        # Apply log transformation to the first feature (Alcohol)
        matrix_copy[:, 0] = np.log(matrix_copy[:, 0] + 1e-9)
        return matrix_copy

    def train_model(self, X_data, y_labels, learning_rate=0.01, total_epochs=100):
        X_processed = self.preprocess_features(X_data)
        X_normalized = self.feature_scaler.fit_transform(X_processed)

        features_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        labels_tensor = torch.tensor(y_labels, dtype=torch.long)

        optimizer_obj = optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        for epoch_idx in range(total_epochs):
            self.network.train()
            optimizer_obj.zero_grad()
            model_outputs = self.network(features_tensor)
            loss_value = loss_function(model_outputs, labels_tensor)
            loss_value.backward()
            optimizer_obj.step()

    def generate_predictions(self, X_input):
        X_processed = self.preprocess_features(X_input)
        X_normalized = self.feature_scaler.transform(X_processed)
        features_tensor = torch.tensor(X_normalized, dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            logits_output = self.network(features_tensor)
            predicted_classes = torch.argmax(logits_output, dim=1)
        
        return predicted_classes.numpy()

    def save_pipeline(self, save_path):
        torch.save(self, save_path)

    @staticmethod
    def load_pipeline(load_path):
        with torch.serialization.safe_globals([NeuralNetPipeline]):
            loaded_instance = torch.load(load_path, weights_only=False)
        return loaded_instance

    @staticmethod
    def search_hyperparams(X_train_data, y_train_labels, X_eval_data, y_eval_labels):
        neuron_counts = [8, 16, 32]
        learning_rates = [0.01, 0.001]
        epoch_counts = [50, 100]
        
        peak_accuracy = 0.0
        optimum_params = None
        best_model_instance = None
        
        print("--- Initiating Hyperparameter Search ---")
        
        param_combinations = product(neuron_counts, learning_rates, epoch_counts)
        for num_neurons, rate_learn, num_epochs in param_combinations:
            current_pipeline = NeuralNetPipeline(neuron_count=num_neurons)
            current_pipeline.train_model(X_train_data, y_train_labels, learning_rate=rate_learn, total_epochs=num_epochs)
            
            predictions_eval = current_pipeline.generate_predictions(X_eval_data)
            current_accuracy = np.mean(predictions_eval == y_eval_labels)
            
            print(f"Neurons: {num_neurons}, LR: {rate_learn}, Epochs: {num_epochs} -> Eval Accuracy: {current_accuracy:.4f}")
            
            if current_accuracy > peak_accuracy:
                peak_accuracy = current_accuracy
                optimum_params = (num_neurons, rate_learn, num_epochs)
                best_model_instance = current_pipeline
                
        print("\n--- Hyperparameter Search Complete ---")
        print(f"Peak Accuracy: {peak_accuracy:.4f}")
        print(f"Optimum Params (Neurons, LR, Epochs): {optimum_params}")
        
        return best_model_instance, optimum_params, peak_accuracy

# ----------------------------
# Train and Evaluate
# ----------------------------
X_training_subset, X_validation_set, y_training_subset, y_validation_set = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

best_model, optimal_settings, validation_accuracy = NeuralNetPipeline.search_hyperparams(
    X_training_subset, y_training_subset, X_validation_set, y_validation_set
)

print("\n--- Assessing Optimal Model on Test Data ---")
test_predictions = best_model.generate_predictions(X_validation_test)
final_test_accuracy = np.mean(test_predictions == y_validation_test)
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")

# Save the best model
file_path_output = "optimal_wine_nn_pipeline.pth"
best_model.save_pipeline(file_path_output)
print(f"\nOptimal pipeline saved to {file_path_output}")

# Load the saved model and test it
reloaded_pipeline = NeuralNetPipeline.load_pipeline(file_path_output)
test_sample = X_validation_test[0].reshape(1, -1)
prediction_output = reloaded_pipeline.generate_predictions(test_sample)

print(f"\nPrediction for sample {X_validation_test[0]}: {prediction_output[0]}")
print(f"Actual value for sample: {y_validation_test[0]}")
