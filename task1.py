
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, accuracy_score
from tune_hyperparameters import tune_hyperparameters
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
    """Returns the most accurate model for diff train splits after c value tuning. Model - Log regression

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
        #clf = LogisticRegression().fit(X_train_train, y_train_train)
        #accuracy = clf.score(X_train_val, y_train_val)

        if accuracy > max_accuracy:
            final_model = clf
            max_accuracy = accuracy

    return clf, max_accuracy


def get_most_fair_model(X_train, y_train):
    """Returns the most fair model for diff train splits after c value tuning. Model - Log regression

        Args: 
            X_train (np.arr): Train Dataset Features
            y_train (np.arr): Train Dataset Labels
        Returns:
            final_model (LogisticRegression()): Model that peforms the best on validation
            max_tp (float): Maximum true positive rate achieved
    """

    max_tp = 0
    final_model = None

    for i in range(5):
        X_train_train, X_train_val, y_train_train, y_train_val = split_train_set(X_train, y_train)   
        model_params, accuracy = tune_hyperparameters(X=X_train_train, y=y_train_train, model=LogisticRegression())
        clf = LogisticRegression(C=model_params['C'], penalty=model_params['penalty'], solver=model_params['solver']).fit(X=X_train_train, y=y_train_train)
        #clf = LogisticRegression().fit(X_train_train, y_train_train)
        
        tp = get_fairness_score(clf, X_train_val, y_train_val)

        if tp > max_tp:
            final_model = clf
            min_tp = tp

    return clf, min_tp

def get_most_fair_accurate_model(X_train, y_train):
    """Returns the most fair and accurate model for diff train splits after c value tuning. Model - Log regression

        Args: 
            X_train (np.arr): Train Dataset Features
            y_train (np.arr): Train Dataset Labels
        Returns:
            final_model (LogisticRegression()): Model that peforms the best on validation
            max_tp (float): Maximum true positive rate achieved
    """
    max_score = 0
    final_model = None

    for i in range(5):
        X_train_train, X_train_val, y_train_train, y_train_val = split_train_set(X_train, y_train)   
        model_params, accuracy = tune_hyperparameters(X=X_train_train, y=y_train_train, model=LogisticRegression())
        clf = LogisticRegression(C=model_params['C'], penalty=model_params['penalty'], solver=model_params['solver']).fit(X=X_train_train, y=y_train_train)
        
        #clf = LogisticRegression().fit(X_train_train, y_train_train)

        tp = get_fairness_score(clf, X_train_val, y_train_val)
        #accuracy = clf.score(X_train_val, y_train_val)
        score = accuracy * tp

        if score > max_score:
            final_model = clf
            max_score = score

    return clf, max_score

def get_fairness_score(model, X_test, y_test):
    """Returns the true positive rate

        Args:
            model (model): Classification model
            X_test (np.arr): Test Dataset Features
            y_test (nparr): Test Dataset Labels
        Returns:
            tp (float): True Positive rate
    """
    df = pd.DataFrame(X_test)
    df['label'] = y_test

    sens_tp_1_df = df[df[4] == 1]
    X_sens_1 = sens_tp_1_df.drop(['label'], axis=1).to_numpy()
    y_sens_1 = sens_tp_1_df['label'].to_numpy()

    sens_tp_2_df = df[df[4] == 2]
    X_sens_2 = sens_tp_2_df.drop(['label'], axis=1).to_numpy()
    y_sens_2 = sens_tp_2_df['label'].to_numpy()

    y_pred_1 = model.predict(X_sens_1)
    y_pred_2 = model.predict(X_sens_2)

    acc_score_1 = model.score(X_sens_1, y_sens_1)
    acc_score_2 = model.score(X_sens_2, y_sens_2)

    return 1 - (acc_score_1 - acc_score_2)

def get_roc_curve(model, X_test, y_test, title=""):
    """Method to plot the ROC Curve 

        Args:
            model (model): Classification model
            X_test (np.arr): Features of Test Dataset
            y_test (np.arr): Labels of Test Dataset
            title (str): Plot title
    """
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.show()


def get_metrics_for_data(features, label, fair_accurate=False):
    """Repeated Operations - need to optimize hyperparameter (whatever that means)"""

    if fair_accurate:
        X_train, X_test, y_train, y_test = split_train_test_set(features, label)
        model, max_score = get_most_fair_accurate_model(X_train, y_train)
        acc = model.score(X_test, y_test)
        tp = get_fairness_score(model, X_test, y_test)
        return model, acc, tp 

    X_train, X_test, y_train, y_test = split_train_test_set(features, label)
    acc_model, max_accuracy = get_most_accurate_model(X_train, y_train)
    acc_model_acc = acc_model.score(X_test, y_test)
    tp = get_fairness_score(acc_model, X_test, y_test)
    acc_model_tp = tp
    X_test_1, y_test_1 = X_test, y_test

    X_train, X_test, y_train, y_test = split_train_test_set(features, label)
    fair_model, max_accuracy = get_most_fair_model(X_train, y_train)
    accuracy = fair_model.score(X_test, y_test)
    fair_model_acc = accuracy
    tp = get_fairness_score(fair_model, X_test, y_test)
    fair_model_tp = tp

    return X_test_1, y_test_1, X_test, y_test, acc_model, fair_model, acc_model_acc, acc_model_tp, fair_model_acc, fair_model_tp

if __name__ == '__main__':
    features, label = np.load("features.npy"), np.load("label.npy")
    X_test_1, y_test_1, X_test, y_test, acc_model, fair_model, acc_model_acc, acc_model_tp, fair_model_acc, fair_model_tp = get_metrics_for_data(features, label)

    print(f"\nAccurate Model Accuracy {acc_model_acc} \n Accurate Model Fairness {acc_model_tp} \n Fair Model Accuracy {fair_model_acc} \n Fair Model Fairness {acc_model_tp} \n")

    #with open('acc_model_task1.pkl','wb') as f:
    #    pickle.dump(acc_model,f)

    #with open('fair_model_task1.pkl','wb') as f:
    #    pickle.dump(fair_model,f)

    #get_roc_curve(X_test_1, y_test_1, "Accurate Model ROC")
    #get_roc_curve(X_test, y_test, "Fair Model ROC")

