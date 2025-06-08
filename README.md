# KNN-Iris-Classifier-AI-Internship
# 🌼 KNN Iris Classifier – AI & ML Internship Task 6

This project implements the **K-Nearest Neighbors (KNN)** classification algorithm using the **Iris dataset**, submitted as part of **Task 6** for the AI & ML Internship. It includes accuracy comparison across K values, confusion matrix evaluation, and 2D decision boundary visualization.

---

## 🎯 Objective

- Understand the working of the KNN algorithm for multi-class classification.
- Analyze model performance for different K values.
- Visualize decision boundaries and accuracy trends.
- Apply feature scaling for distance-based algorithms.

---

## 🧠 Concepts Covered

- Instance-based Learning
- Feature Normalization (StandardScaler)
- K Value Selection
- Accuracy & Confusion Matrix
- Decision Boundary Visualization (2D)
- Euclidean Distance Metric

---

## 🔧 Technologies Used

- Python 3
- Scikit-learn
- Matplotlib
- NumPy
- Git & GitHub

---

## 📁 Project Structure

```
KNN-Iris-Classifier-AI-Internship/
├── knn_classifier.py
├── accuracy_plot.png
├── decision_boundary_plot.png
└── README.md
```

---

## 📊 Results

- ✅ **Best K value**: `3`
- ✅ **Accuracy**: `100%` on test dataset
- ✅ **Confusion Matrix**:
  ```
  [[19  0  0]
   [ 0 13  0]
   [ 0  0 13]]
  ```

- 📈 Accuracy plot: saved as `accuracy_plot.png`
- 🌈 Decision boundary: saved as `decision_boundary_plot.png`

---

## 📚 Dataset

- **Iris Dataset**
  - Features: Sepal length, Sepal width, Petal length, Petal width
  - Classes: Setosa, Versicolor, Virginica
  - Source: [https://www.kaggle.com/datasets/uciml/iris](https://www.kaggle.com/datasets/uciml/iris)

---

## 📤 Submission Info

- 🔖 **Task**: Task 6 – KNN Classifier
- 👩‍💻 **Intern**: Sukavarshini
- 📬 **Email**: varshini8754@gmail.com
- 🔗 **GitHub Repo**: [https://github.com/Sukavarshini/KNN-Iris-Classifier-AI-Internship](https://github.com/Sukavarshini/KNN-Iris-Classifier-AI-Internship)

---

## 💬 Key Questions Answered

**Q: Why is normalization important in KNN?**  
KNN uses distance to determine neighbors. Without normalization, features with large scales dominate.

**Q: How do you choose the right K?**  
Try multiple K values (e.g., 1 to 10), plot accuracy, and pick the one with the best performance.

**Q: Is KNN sensitive to noise?**  
Yes. A small K value may be sensitive to outliers and noise.

**Q: Does KNN handle multi-class problems?**  
Yes. It predicts the majority class among the K nearest neighbors, even for more than 2 classes.

---

## ✅ Conclusion

This task helped reinforce the importance of data preprocessing in distance-based algorithms and provided visual insights into how KNN performs on simple datasets like Iris.



---
