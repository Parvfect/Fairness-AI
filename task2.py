
import numpy as np
import pandas as pd
from task1 import get_metrics_for_data
import pickle


def reweigh_fairness():
    """Reweighs the weight for fairness of the sensitive group"""
    data = pd.DataFrame(features, columns = ACSEmployment.features)
    data['label'] = label
    favorable_classes = [True]
    protected_attribute_names = [ACSEmployment.group]
    privileged_classes = np.array([[1]])


    # Spec for Ai360
    data_for_aif = StandardDataset(data, 'label', favorable_classes = favorable_classes,
    protected_attribute_names = protected_attribute_names,
    privileged_classes = privileged_classes)
    privileged_groups = [{'DIS': 1}]
    unprivileged_groups = [{'DIS': 2}]

    new_weights = Reweighing(unprivileged_groups, privileged_groups).transform(data_for_aif)
    df, df1 = new_weights.convert_to_dataframe()

    features = df.drop(['label'], axis=1).to_numpy()
    label = df['label'].to_numpy()

    return features, label

features = np.load("fair_features.npy")
label = np.load("fair_label.npy")

X_test_1, y_test_1, X_test, y_test, acc_model, fair_model, acc_model_acc, acc_model_tp, fair_model_acc, fair_model_tp = get_metrics_for_data(features, label)

print(f"\nAccurate Model Accuracy {acc_model_acc} \n Accurate Model Fairness {acc_model_tp} \n Fair Model Accuracy {fair_model_acc} \n Fair Model Fairness {acc_model_tp} \n")

with open('acc_model_task2.pkl','wb') as f:
    pickle.dump(acc_model,f)

with open('fair_model_task2.pkl','wb') as f:
    pickle.dump(fair_model,f)

#get_roc_curve(X_test_1, y_test_1, "Accurate Model ROC")
#get_roc_curve(X_test, y_test, "Fair Model ROC")

