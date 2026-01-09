from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_true = np.load("y_true.npy")
y_pred = np.load("y_pred.npy")

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
