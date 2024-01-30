
import numpy as np
import pandas as pd
from task1 import get_metrics_for_data

features = np.load("fair_features.npy")
label = np.load("fair_label.npy")

model, accuracy, tp = get_metrics_for_data(features, label)
X_test_1, y_test_1, X_test, y_test, acc_model, fair_model, acc_model_acc, acc_model_tp, fair_model_acc, fair_model_tp = get_metrics_for_data(features, label)

print(f"\nAccurate Model Accuracy {acc_model_acc} \n Accurate Model Fairness {acc_model_tp} \n Fair Model Accuracy {fair_model_acc} \n Fair Model Fairness {acc_model_tp} \n")

with open('acc_model_task2.pkl','wb') as f:
    pickle.dump(acc_model,f)

with open('fair_model_task2.pkl','wb') as f:
    pickle.dump(fair_model,f)

get_roc_curve(X_test_1, y_test_1, "Accurate Model ROC")
get_roc_curve(X_test, y_test, "Fair Model ROC")

