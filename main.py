#!/usr/bin/env python
# coding: utf-8

#Define constants
from pipeline import *
import pandas as pd
import os
import time

#Define constants

DATA_DIR = "data"
DATA_FILE = "projects_2012_2013.csv"
RESULTS_DIR = "results"
RESULTS_FILE = "final_results.csv"

CLEAN_PARAMETERS = {'years_range': [12, 14], 'sd_threshold': 3,\
                    'how_fill_nan':'median'}
SPLIT_PARAMETERS = {
                    'N_SAMPLES': 3,
                    'TRAIN_START': '2012-01-01',
                    'TRAIN_DAYS': 122,
                    'GAP': 60
                    }
VARIABLES = {
             'LABEL' : 'label',
             'TEMPORAL_VAL_VAR' : 'date_posted',
             'DATE_OUTCOME' : 'datefullyfunded',
             'IDENTIFICATION_VARS' : ['projectid', 'date_posted'],
             'FLAG_VARS' : ['school_charter', 'school_magnet', 
                            'eligible_double_your_impact_match'],
             'CONTINUOUS_VARS' : ['total_price_including_optional_support', 
                                  'students_reached'],
             'CATEGORICAL_VARS' : ['school_state', 'secondary_focus_subject',\
                                   'primary_focus_area', 'teacher_prefix',\
                                   'secondary_focus_area', 'resource_type',\
                                   'poverty_level', 'grade_level',\
                                   'primary_focus_subject'],
             'VARS_TO_DROP' : ['teacher_acctid', 'schoolid', 'school_ncesid',\
                               'school_latitude', 'school_longitude',\
                               'school_city', 'school_metro', 'school_county',\
                               'school_district', 'datefullyfunded'],
            }

'''
#Small grid for testing
CUSTOM_GRID =  {'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
                'KNN' :{'n_neighbors': [5],'weights': ['uniform'],
                        'algorithm': ['auto']},
                'DT': {'criterion': ['gini'], 'max_depth': [1],
                       'min_samples_split': [10]},
                'SVM' :{'random_state':[0], 'tol':[1e-5]},
                'RF':{'n_estimators': [1], 'max_depth': [1], 
                      'max_features': ['sqrt'],'min_samples_split': [10]},
                'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
                'BA': {'base_estimator': [LogisticRegression()], 
                       'n_estimators':[1]}}
'''
CUSTOM_GRID = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100],
          'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 
          'n_jobs': [-1]},
    'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'AB': {'algorithm': ['SAMME', 'SAMME.R'], 
           'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],
           'min_samples_split': [2,5,10]},
    'SVM':{'random_state':[0], 'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],
           'tol':[1e-5]},
    'KNN':{'n_neighbors': [1,10,100],'weights': ['uniform','distance'],
           'algorithm': ['auto','ball_tree','kd_tree']},
     'BA': {'base_estimator': [LogisticRegression()], "n_estimators":[1]}}

EVAL_METRICS_BY_LEVEL = (['accuracy', 'precision', 'recall', 'f1'],\
                         [1,2,5,10,20,30,50])
EVAL_METRICS = ['auc']
MODELS = [ 'SVM', 'LR', 'DT','KNN', 'RF', 'AB', 'BA']


def main(data_dir=DATA_DIR, file=DATA_FILE, results_dir=RESULTS_DIR, 
         results_file=RESULTS_FILE, variables=VARIABLES, 
         split_parameters=SPLIT_PARAMETERS, clean_parameters=CLEAN_PARAMETERS):
    
    sample = 1
    label = variables['LABEL']
    
    while sample <= split_parameters['N_SAMPLES']:
        
        test_csv = os.path.join(data_dir, "{}_test_{}.csv".format(file[:-4], sample))
        train_csv = os.path.join(data_dir, "{}_train_{}.csv".format(file[:-4], sample))
        
        #If the csv with the sampel data tu use was already created, cleaned, 
        #etc. then just open it:
        if os.path.exists(test_csv) and os.path.exists(train_csv):
            df_test = pd.read_csv(test_csv) 
            df_train = pd.read_csv(train_csv) 
        
        #If the csv with the sampel data tu use was not created then use the 
        #original data csv file, split it into training and testing, clean it,
        #create labels and create features: 
        else:
            df = get_csv(data_dir, file)
           
            temp_val_date = variables['TEMPORAL_VAL_VAR']
            outcome_date = variables['DATE_OUTCOME']
            df = to_date(df, [temp_val_date, outcome_date], 
                         clean_parameters['years_range'])
           
            df[label] = ((df[outcome_date] - df[temp_val_date]) > pd.to_timedelta(60,\
                                                                  unit='d')).astype(int)
           
            GAP = split_parameters['GAP']
            
            #I decided to split into 3 samples. The first use 4 month for training 
            #2 for the oucome to happen and 4 to test in the, then 10, 2, 4 and 
            #finally 16, 2, and 4 respectively. 
            df_train, df_test = split_train_test_over_time(df, 
                                temp_val_date, 
                                split_parameters['TRAIN_START'], 
                                (split_parameters['TRAIN_DAYS']+GAP) * sample - GAP, 
                                GAP)

            df_train = categorical_to_dummy(df_train, variables['CATEGORICAL_VARS'])
            df_test = categorical_to_dummy(df_test, variables['CATEGORICAL_VARS'])
            
            df_train = flag_to_dummy(df_train, variables['FLAG_VARS'], rename=False)
            df_test = flag_to_dummy(df_test, variables['FLAG_VARS'], rename=False)

            df_test = df_test.loc[:,[x for x in df_test.columns if x not in variables['VARS_TO_DROP']]]
            df_train = df_train.loc[:,[x for x in df_train.columns if x not in variables['VARS_TO_DROP']]]
            
            df_test = remove_outliers(df_test, variables['CONTINUOUS_VARS'],
                                      clean_parameters['sd_threshold'])
            df_train = remove_outliers(df_train, variables['CONTINUOUS_VARS'],
                                       clean_parameters['sd_threshold'])
            
            fill_nan(df_test, variables['CONTINUOUS_VARS'], clean_parameters['how_fill_nan'])
            fill_nan(df_train, variables['CONTINUOUS_VARS'], clean_parameters['how_fill_nan'])
            
            df_test = discretize_variable(df_test, variables['CONTINUOUS_VARS'])
            df_train = discretize_variable(df_train, variables['CONTINUOUS_VARS'])
            
            #Save the tranformed sample data into two csv files
            df_test.to_csv(test_csv, index=False)
            df_train.to_csv(train_csv, index=False)
        
        #Create a list with the name of the attributes I want to use in the models
        attributes_lst = list(df_train.columns)
        attributes_lst.remove(label)
        for var in variables['IDENTIFICATION_VARS']+variables['CONTINUOUS_VARS']:
            attributes_lst.remove(var)

        results = classify(df_train, df_test, label, MODELS, EVAL_METRICS, 
                           EVAL_METRICS_BY_LEVEL, CUSTOM_GRID, attributes_lst)
        results['sample'] = sample
        
        #Save the results of the metrics evaluated when running each model into a csv
        if sample == 1:
            results.to_csv(os.path.join(results_dir, results_file), index=False)
        else:
            with open(os.path.join(results_dir, results_file), 'a') as f:
                results.to_csv(f, header=False, index=False) 

        sample += 1
        


if __name__ == "__main__":
  main()

