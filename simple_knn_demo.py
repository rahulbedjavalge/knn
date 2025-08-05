"""
Simple KNN Example - Quick Demo
This script provides a simple demonstration of KNN classification and regression.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt

def simple_knn_classification():
    """Simple KNN classification example."""
    print("=== KNN Classification Example ===")
    
    # Generate dummy dataset
    X, y = make_classification(n_samples=500, n_features=4, n_classes=3, 
                             n_informative=3, n_redundant=1, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train KNN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn_clf.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Visualize the results (using first 2 features for visualization)
    plt.figure(figsize=(12, 5))
    
    # Plot training data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
    plt.title('Training Data (First 2 Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    # Plot test predictions
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.title(f'Test Predictions (Accuracy: {accuracy:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig('simple_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy

def simple_knn_regression():
    """Simple KNN regression example."""
    print("\n=== KNN Regression Example ===")
    
    # Generate dummy dataset
    X, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=42)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create and train KNN regressor
    knn_reg = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_reg.predict(X_test)
    
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    
    print(f"Dataset shape: {X.shape}")
    print(f"R² Score: {r2:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    
    # Sort for better visualization
    sort_idx = np.argsort(X_test.flatten())
    
    plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', color='blue')
    plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', color='red')
    plt.plot(X_test[sort_idx], y_pred[sort_idx], color='green', linewidth=2, 
             label=f'KNN Predictions (R²={r2:.3f})')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('KNN Regression Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('simple_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return r2

def compare_k_values():
    """Compare different k values for KNN."""
    print("\n=== Comparing Different K Values ===")
    
    # Generate classification dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                             n_informative=3, n_redundant=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different k values
    k_values = range(1, 21)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"k={k:2d}: Accuracy = {accuracy:.4f}")
    
    # Find optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    
    print(f"\nOptimal k: {optimal_k} with accuracy: {best_accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=6)
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
               label=f'Optimal k={optimal_k}')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title('KNN Classification: Effect of k Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('k_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_k, best_accuracy

def main():
    """Main function to run all examples."""
    print("KNN Machine Learning Examples with Dummy Data")
    print("=" * 50)
    
    # Run classification example
    clf_accuracy = simple_knn_classification()
    
    # Run regression example
    reg_r2 = simple_knn_regression()
    
    # Compare k values
    optimal_k, best_accuracy = compare_k_values()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Classification Accuracy: {clf_accuracy:.4f}")
    print(f"Regression R² Score: {reg_r2:.4f}")
    print(f"Optimal k Value: {optimal_k}")
    print(f"Best Accuracy with Optimal k: {best_accuracy:.4f}")
    print("\nAll plots saved to current directory")

if __name__ == "__main__":
    main()
