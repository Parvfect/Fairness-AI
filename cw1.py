# Testing variation of Hyperparameter for Regression, reweighing using fairness methods and using appropiate model selection for the final model which is both accurate and fair


import folktables
from folktables import ACSDataSource
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tune_hyperparameters import tune_hyperparameters

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


def split_train_test_set(features, label, test_size=0.3):
    """Splits the data into train and test sets based on split 

        Args:
            data (df): DataFrame containing the data
            test_size (float): Proportional size of the test dataset
        Returns:
            train (arr): Train numpy array
            test  (arr): Test numpy array  
    """ 
    return train_test_split(features, label, test_size=test_size, shuffle=True)

def split_train_set(X_train, y_train, val_size=0.2, shuffle_train=False):
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
        X_train, X_test, y_train, y_test = split_train_test_set(X_train, y_train)
        train_train, train_val = split_train_test_set(X_train, y_train, test_size=0.2)
        return train_train, train_val, test
    return split_train_test_set(X_train, y_train, test_size=0.2)



def get_most_accurate_model(X_train, y_train):
    """Returns the most accurate model for diff train splits. Model - Log regression

        Args: 
            X_train (np.arr): Train Dataset Features
            y_train (np.arr): Train Dataset Labels
        Returns:
            final_model (LogisticRegression()): Model that peforms the best on validation
            max_accuracy (float): Maximum accuracy achieved
    """

    max_accuracy = 0.0
    final_model = None

    for i in range(5):
        X_train_train, X_train_val, y_train_train, y_train_val = split_train_set(X_train, y_train)   
        model_params, accuracy = tune_hyperparameters(X=X_train_train, y=y_train_train, model=LogisticRegression())
        clf = LogisticRegression(C=model_params['C'], penalty=model_params['penalty'], solver=model_params['solver']).fit(X=X_train_train, y=y_train_train)
        
        if accuracy > max_accuracy:
            final_model = clf
            max_accuracy = accuracy

    return clf, max_accuracy

def get_fairness_score(model, X_test, y_test):

    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return tp

def get_metrics_for_data(features, label):
    """Repeated Operations - need to optimize hyperparameter (whatever that means)"""

    X_train, X_test, y_train, y_test = split_train_test_set(features, label)
    model, max_accuracy = get_most_accurate_model(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model score {model.score(X_test, y_test)}")
    tp = get_fairness_score(model, X_test, y_test)/len([i for i in y_test if i ==1])
    print(f"Model fairness {tp}")

    return model, accuracy, tp

model, accuracy, tp = get_metrics_for_data(features, label)

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