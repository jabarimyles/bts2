#-- base packages
import os
import sys
import pickle

#-- pypi packages
import pandas as pd
import lightgbm 
from bayes_opt import BayesianOptimization
import pdb


#-- custom packages
from train_model import logistic
from train_model import tune_lgbm
from train_model import lgbm
from createModelingData import get_modeling_data



if __name__ == '__main__':
    models_dir = './models'
    dup_model_app = '___'

    model_name = sys.argv[1]
    model_names = [i.split(dup_model_app)[0] for i in os.listdir(models_dir)]
    conflicts = [i for i in model_names if model_name == i]
    num_conflicts = len(conflicts)
    if num_conflicts > 0:
        model_name = model_name + dup_model_app + str(num_conflicts + 1)
        print('model name already taken... new model renamed to {}'.format(model_name))
    full_model_path = os.path.join(models_dir, model_name)
    os.mkdir(full_model_path)

    x_train, y_train, x_test, y_test = get_modeling_data()
    x_train.to_csv(os.path.join(full_model_path, 'x_train.csv'), index=False)
    y_train.to_csv(os.path.join(full_model_path, 'y_train.csv'), index=False)
    x_test.to_csv(os.path.join(full_model_path, 'x_test.csv'), index=False)
    y_test.to_csv(os.path.join(full_model_path, 'y_test.csv'), index=False)

    id_vars = ['game_date', 'game_pk', 'batter', 'starting_pitcher', 'ABs', 'hits', 'hit_ind']
    opt_params, clf = tune_lgbm(x_train.drop(id_vars, axis=1), y_train, x_test.drop(id_vars, axis=1), y_test)
    print(opt_params)
    
    model = lgbm(x_train.drop(id_vars, axis=1), y_train, x_test.drop(id_vars, axis=1), y_test, opt_params, clf)
    #model = logistic(x_train.drop(id_vars, axis=1), y_train)

    model_name_file = model_name + '.pickle'
    model_pkl_fp = os.path.join(full_model_path, model_name_file)
    pickle.dump(model, open(model_pkl_fp, 'wb'))
