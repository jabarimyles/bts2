#-- base packages
import os
import sys



#-- Pypi stats models
import numpy as np
import pandas as pd
import pdb

#import matplotlib.pyplot as plt
#import statsmodels.api as sm
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (train_test_split,
                                     cross_validate,
                                     GridSearchCV,
                                     RandomizedSearchCV
                                    )
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import precision_score, make_scorer





def logistic(x_train, y_train):

    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   n_jobs=None, penalty='l1',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

    model.fit(x_train, y_train)
    
    return model#, modeling_data[inputs], l1_scores

def tune_lgbm(x_train, y_train, x_test, y_test):
    #Create dataset that can go directly to model
    train_data = lightgbm.Dataset(data=x_train, label=y_train, free_raw_data=False)

    scorer = make_scorer(precision_score)

    fit_params={"eval_metric" : 'auc',
            "eval_set" : [(x_test,y_test)],
            'eval_names': ['test'],
            'verbose': 100,
            'categorical_feature': 'auto',
            'early_stopping_rounds': 100}
    
    param_test ={'num_leaves': sp_randint(6, 100), 
             'min_child_samples': sp_randint(100, 2000), # i.e. min data in leaf
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             #'subsample': sp_uniform(loc=0.2, scale=0.8), #bagging_fraction
             #'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             #'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             #'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
             #'learning_rate': [.0001,.001 ,.005, .01, .05],
             'max_depth': sp_randint(10,50)}
    
    

    #n_estimators is set to a "large value". The actual number of trees build will depend on early stopping 
    clf = lightgbm.LGBMClassifier(random_state=925, silent=True, n_jobs=4, n_estimators=1000)
    
    gs = RandomizedSearchCV(
        estimator=clf, param_distributions=param_test, 
        n_iter=20, 
        scoring="accuracy",
        cv=3,
        refit=True,
        random_state=925,
        verbose=True)
    
    
    results = gs.fit(x_train, y_train, **fit_params) #

    print(gs.best_params_, gs.best_score_)

    return gs.best_params_, clf


def lgbm(x_train, y_train, x_test, y_test, opt_params, clf):
    
    categorical_features = [c for c, col in enumerate(x_train.columns) if 'cat' in col]

    train_data = lightgbm.Dataset(x_train, label=y_train, categorical_feature=categorical_features)

    test_data = lightgbm.Dataset(x_test, label=y_test)
    


    model = lightgbm.train(params=opt_params,
                        train_set=train_data,
                       valid_sets=test_data,
                       num_boost_round=1000, 
                       random_state=925)
    

    final_model = lightgbm.LGBMClassifier(**clf.get_params())

    final_model.set_params(**opt_params)

    final_model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

    return final_model



def random_forrest():

    inputs = [

        'rp_BA', 'rp_AB_div_PA', 'ytd_BA', 'ytd_AB_div_PA', 'rp_BA_sp',
        'rp_AB_div_PA_sp', 'ytd_BA_sp', 'ytd_AB_div_PA_sp', 'Bot',
        'L-L', 'L-R', 'R-L',
        'rp_hits_var', 'ytd_hits_var',
        'match_year_PAs', 'match_year_BA', 'match_year_AB_div_PA',
        'match_career_PAs', 'match_career_BA', 'match_career_AB_div_PA'
    ]

    pipe = Pipeline([
        ('scale', MinMaxScaler()),
        ('m', RandomForestClassifier())
    ])
    param_grid = {
        'm__n_estimators': [30, 40, 50], # num trees
        'm__max_depth': [14, 22, 30], #max depth of trees
        'm__min_samples_split': [8, 20, 100], #min samples per leaf
        'm__max_features': [5, 10, 15], # num features
        'm__random_state': [33],
    }
    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring='f1_weighted',
                        n_jobs=-1,
                        cv=5
                       )
    grid.fit(train[inputs], train['hit_ind'])
    model = grid.best_estimator_
    cv_results = cross_validate(model,
                                train[inputs],
                                train['hit_ind'],
                                cv=10,
                                n_jobs=-1,
                                scoring='f1_weighted',
                                return_train_score=True
                               )
    cv_results = pd.DataFrame(cv_results)
    # %% validate
    scores_out = model.score(test[inputs], test['hit_ind'])
    y_preds = model.predict(test[inputs])
    y_pred_probs = model.predict_proba(test[inputs])

    prec, recall, thresholds = precision_recall_curve(test['hit_ind'], y_pred_probs[:,1])

    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.show()

    print(str(round(scores_out*100)) + '%' + ' accuracury')
    print(confusion_matrix(test['hit_ind'], y_preds))

    return model, modeling_data[inputs]
    