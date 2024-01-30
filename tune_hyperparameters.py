# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/


# example of grid searching key hyperparametres for logistic regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import sys

def tune_hyperparameters(X, y, model):

    solvers = ['lbfgs']
    penalty = ['l2']
    c_values = [1000, 100, 10, 1.0, 0.1, 0.01]

    # define grid search
    grid = dict(solver=solvers, penalty=penalty,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    
    grid_result = grid_search.fit(X, y)
    return grid_result.best_params_, grid_result.best_score_