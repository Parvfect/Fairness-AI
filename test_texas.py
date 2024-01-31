
import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from task1 import get_fairness_score
from aif360.algorithms.preprocessing import Reweighing
import pickle

#(Age) must be greater than 16 and less than 90, and (Person weight) must be greater than or equal to 1

def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    df = df[df['PWGTP'] >= 1]
    return df


ACSEmployment = folktables.BasicProblem(
features=[
'AGEP', #age; for range of values of features please check Appendix B.4 of Retiring Adult: New Datasets for Fair Machine Learning NeurIPS 2021 paper
'SCHL', #educational attainment
'MAR', #marital status
'RELP', #relationship
'DIS', #disability recode
'ESP', #employment status of parents
'CIT', #citizenship status
'MIG', #mobility status (lived here 1 year ago)
'MIL', #military service
'ANC', #ancestry recode
'NATIVITY', #nativity
'DEAR', #hearing difficulty
'DEYE', #vision difficulty
'DREM', #cognitive difficulty
'SEX', #sex
'RAC1P', #recoded detailed race code
'GCL', #grandparents living with grandchildren
],
target='ESR', #employment status recode
target_transform=lambda x: x == 1,
group='DIS',
preprocess=employment_filter,
postprocess=lambda x: np.nan_to_num(x, -1),
)

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["TX"], download=True) #data for Florida state
features, label, group = ACSEmployment.df_to_numpy(acs_data)

model_1 = pickle.load(open('acc_model_task1.pkl', 'rb'))
model_2 = pickle.load(open('fair_model_task1.pkl', 'rb'))
model_3 = pickle.load(open('acc_model_task2.pkl', 'rb'))
model_4 = pickle.load(open('fair_model_task2.pkl', 'rb'))
model_5 = pickle.load(open('model_1_task3.pkl', 'rb'))
model_6 = pickle.load(open('model_2_task3.pkl', 'rb'))


print("Accuracy for Texas")
print(model_1.score(features, label))
print(model_2.score(features, label))
print(model_3.score(features, label))
print(model_4.score(features, label))
print(model_5.score(features, label))
print(model_6.score(features, label))

print("Fairness for Texas")
print(get_fairness_score(model_1, features, label))
print(get_fairness_score(model_2, features, label))
print(get_fairness_score(model_3, features, label))
print(get_fairness_score(model_4, features, label))
print(get_fairness_score(model_5, features, label))
print(get_fairness_score(model_6, features, label))