# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:35:08 2018

@author: Manoj Kumar Singh manoj@wustl.edu
"""
## Two pound sign (##) indicate comment lines.
## One pound sign (#) indicates code lines. Code lines can be made active by removing
## pound character form the begingin of the line.

## This tutorial uses a dataset gathered from direct marketing campaigns 
## (phone calls) of a Portuguese banking institution.
## "S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the 
## Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, 
## June 2014"
## http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

## First module of this tutorial covers basics of pandas dataframe. We will 
## readin data to a dateframe and will print its charactersistics. 

## Read csv file to a dataframe.
df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')

## Print first 5 rows of dataframe.
#print( df.head() ) # try df.tail()

## Print column names of dataframe.
#print( df.columns )

##Print size of dataframe
#print( df.shape ) ## (rows, columns)


## In second part, we will do basics statistics with columns of dataframe
## Plot histogram of column 'cons.price.idx'
#df.hist(column='cons.price.idx')
#plt.show()

## Calculate mean, standard deviation etc.
#print( df['cons.price.idx'].mean() )

## In thrid part, we will prepare dataset for random forest.
## We will first convert string values to digits using factorize tool.

columns_to_factorize = ['job', 'marital', 'education', 'default', 'housing', 'loan', \
       'contact', 'month', 'day_of_week', 'poutcome', ]

stacked = df[columns_to_factorize].stack()
df[columns_to_factorize] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
#print( df.head() )

## In fourth part, we will 10-fold cross-validation with random forest.
## Add classifier value, yes = 1; no = -1.
df['flag_value'] = 0 #initilalize
df['flag_value'] = np.where( df['y'] == 'yes', 1, df['flag_value'] )
df['flag_value'] = np.where( df['y'] == 'no', -1, df['flag_value'] )


## Normalize data
normData = False ## True or False

if normData:
    ## Count number of 'yes' and 'no' for 'Y' column.
    flag_value_count = df['flag_value'].value_counts()
    ## Count number of "yes" in subscription.
    num_yes = pd.Series(flag_value_count)[1]
    sys.stdout.write('Number of Yes: '+str(num_yes)+'\n')
    #
    num_no = pd.Series(flag_value_count)[-1]
    sys.stdout.write('Number of No: '+str(num_no)+'\n')
    #
    ## Now, normalize data by randomly selecting and downsampling by setting flag value to 0.
    if num_yes > num_no: # if more yes than no
        idx = df.index[(df['flag_value'] == 1)]
        df.loc[(np.random.choice(idx, size=(num_yes-num_no), replace=False)), 'flag_value'] = 0
        del idx
    # 
    if num_no > num_yes: # if more no than yes
        idx = df.index[(df['flag_value'] == -1)]
        df.loc[(np.random.choice(idx, size=(num_no-num_yes), replace=False)), 'flag_value'] = 0
        del idx
    
    ## Count yes and no again.
    flag_value_count = df['flag_value'].value_counts()
    ## Count number of "yes" in subscription.
    num_yes = pd.Series(flag_value_count)[1]
    sys.stdout.write('Number of Yes after norm: '+str(num_yes)+'\n')
    #
    num_no = pd.Series(flag_value_count)[-1]
    sys.stdout.write('Number of No after norm: '+str(num_no)+'\n')
    #
    ## Select new dataframe and reset index
    df = ( df[(df.flag_value==1) | (df.flag_value==-1)] ).reset_index()
    print( df.shape )
    

## We will add another column ['clf_flag'] to mark test, train.
df['clf_flag'] = 'null'
print( df.head() )

## -------------------------
## We will now run 10-fold cross validation with random-forest

## Name of the columns which we want to use as feature for random forest.
feature_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', \
                   'loan','contact', 'month', 'day_of_week', \
                   'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', \
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

## It is better to exclude 'duration' column for a realistic predictive model.

## We will now add columns for value prediction.
df['prob_yes'] = 0.0 # Probability of 'yes'
df['prob_no'] = 0.0 # Probability of 'no'
df['pred_value'] = 0.0 # Predicted value



## Setup 10-fold split. for K-Fold validation.
kf = KFold(n_splits=10, shuffle=True) # 10 fold
kf.get_n_splits( df.index )

## Loop over split index
for train_index, test_index in kf.split( df.index ):
    #
    df.loc[train_index, 'clf_flag'] = 'train'
    #
    df.loc[test_index, 'clf_flag'] = 'test'
    #
    ## Select train and test sets.
    df_train = df[ df['clf_flag']=='train' ]
    df_test = df[ df['clf_flag']=='test' ]

    ## Select target values (Y)
    y_train = df_train['flag_value'].values
    #
    ## Setup random forest classifier and fit to it.
    clf = RandomForestClassifier(n_estimators=101, n_jobs=1)
    clf.fit(df_train[feature_columns], y_train)

    ## Performance Evaluation
    y_test_prob = clf.predict_proba(df_test[feature_columns])
    y_test_pred = clf.predict(df_test[feature_columns])
    #
    for i,idx in enumerate(df_test[feature_columns].index):
        df.loc[idx, 'pred_value'] = y_test_pred[i]
        df.loc[idx, 'prob_no'] = y_test_prob[i][0]
        df.loc[idx, 'prob_yes'] = y_test_prob[i][1]


## Clean dataframe and print output to a csv file
df = df.drop( ['clf_flag'] , axis=1)
df.to_csv('test_out.csv', sep=',')


## Pot ROC curve
y_obs = df['flag_value'].tolist()
y_prob = df['prob_yes'].tolist()

## Calculate TPR, FPR.
fpr_rf, tpr_rf, _ = roc_curve(y_obs, y_prob)

## Calculate AUC
roc_auc = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
