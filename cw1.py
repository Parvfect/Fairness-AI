# Testing variation of Hyperparameter for Regression, reweighing using fairness methods and using appropiate model selection for the final model which is both accurate and fair


import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
acs_data = data_source.get_data(states=["FL"], download=True) #data for Florida state
features, label, group = ACSEmployment.df_to_numpy(acs_data)

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

def split_train_test_set(data, test_size=0.3):
    """Splits the data into train and test sets based on split 

        Args:
            data (df): DataFrame containing the data
            test_size (float): Proportional size of the test dataset
        Returns:
            train (arr): Train numpy array
            test  (arr): Test numpy array  
    """ 
    return train_test_split(data, test_size=test_size, shuffle=True)

def split_train_set(train, val_size=0.2, shuffle_train=False):
    """Splits train set into train-train and train-val sets
        
        Args:
            train (np.arr): Train array
            val_size (float): Proportion of the Validation Set
            shuffle_train (boolean): If true, train is original df, redo the split
        Returns:
            train_train (np.arr): Train Train Set
            train_val (np.arr): Train Val Set
            test (nparr): Only if shuffle_train is True
    """

    if shuffle_train:
        train, test = split_train_test_set(train)
        train_train, train_val = split_train_test_set(train, test_size=0.2)
        return train_train, train_val, test
    return split_train_test_set(train, test_size=0.2)



def test_logistic_regression()

train_train, train_val, test = split_train_set(data, shuffle_train=True)
for i in in range(5):
    print("Seperate the test, train on train train train val, get the one with highest acc compute its fairness and then think about warying hyperparameter")
    print("Then do the reweighing using fairness methods and repeat the procedure")
    print("Then come up with some criterion for model selection based on results and reading and repeat")
    print("Then repeat this on the Texas data")