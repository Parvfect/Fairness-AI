
import numpy as np
import pandas as pd
from task1 import get_metrics_for_data
import pickle

features, label = np.load("features.npy"), np.load("label.npy")
model_1, acc, tp  = get_metrics_for_data(features, label, fair_accurate=True)
print(f"\n Standard Model Accuracy {acc} \n Standard Model Fairness {tp} \n ")

features, label = np.load("fair_features.npy"), np.load("fair_label.npy")

model_2, acc, tp  = get_metrics_for_data(features, label, fair_accurate=True)
print(f"\n Fair Model Accuracy {acc} \n Fair Model Fairness {tp} \n ")

with open('model_1_task3.pkl','wb') as f:
    pickle.dump(model_1,f)

with open('model_2_task3.pkl','wb') as f:
    pickle.dump(model_2,f)
