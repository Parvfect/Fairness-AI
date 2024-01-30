
import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["FL"], download=True) #data for Florida state
features, label, group = ACSEmployment.df_to_numpy(acs_data)
np.save("features.npy", features)
np.save("label.npy", label)
np.save("group.npy", group)

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

model, accuracy, tp = get_metrics_for_data(features, label)