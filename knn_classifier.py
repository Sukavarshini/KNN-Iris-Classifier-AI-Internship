import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap
import os

def load_and_prepare_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, target_names

def evaluate_k_values(X_train, X_test, y_train, y_test, max_k=10):
    print("📊 Accuracy comparison for K values:")
    accuracies = []
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"K={k}: Accuracy = {acc:.2f}")
    return accuracies

def save_accuracy_plot(accuracies, filename="accuracy_vs_k.png"):
    plt.figure()
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='--', color='blue')
    plt.title("Accuracy vs. K value")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"📁 Saved plot: {filename}")

def train_best_model(X_train, X_test, y_train, y_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n✅ Best K = {k}")
    print(f"🎯 Accuracy = {acc:.2f}")
    print("🧩 Confusion Matrix:\n", cm)
    return model, acc, cm

def plot_decision_boundary(X, y, feature_names, k=3, filename="decision_boundary_plot.png"):
    X_vis = X[:, :2]
    X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_vis, y_train_vis)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=ListedColormap(('red', 'green', 'blue')), edgecolor='k')
    plt.title("KNN Decision Boundary (Using 2 Features)")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"📁 Saved plot: {filename}")

def main():
    try:
        # Prepare data
        X, y, feature_names, target_names = load_and_prepare_data()

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Evaluate K values
        accuracies = evaluate_k_values(X_train, X_test, y_train, y_test, max_k=10)
        save_accuracy_plot(accuracies)

        # Train model with best K (manual/auto selection)
        best_k = accuracies.index(max(accuracies)) + 1
        _, _, _ = train_best_model(X_train, X_test, y_train, y_test, k=best_k)

        # Plot decision boundary
        plot_decision_boundary(X, y, feature_names, k=best_k)

        print("\n✅ Task completed successfully.")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
