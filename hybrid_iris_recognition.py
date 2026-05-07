import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Phase 1: Feature Fusion (EfficientNet-B1 + LBP)
X_final_train = np.hstack((F_train, X_train_lbp))
X_final_test = np.hstack((F_test, X_test_lbp))

# Phase 2: Feature Scaling & Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_final_train)
X_test_scaled = scaler.transform(X_final_test)

# Phase 3: Final Master Classifier (RBF-SVM)
final_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
final_svm.fit(X_train_scaled, y_train)

# Phase 4: Evaluation and Scoring
score = final_svm.score(X_test_scaled, y_test)
print("-" * 40)
print(f"Final Proposed Hybrid Accuracy: {score * 100:.2%}")
print("-" * 40)

# Phase 5: Generating Research Metrics
y_pred = final_svm.predict(X_test_scaled)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))