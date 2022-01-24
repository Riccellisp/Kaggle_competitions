# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 18:11:48 2022

@author: bruno.silva
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import math
from random import choice
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

def random_choice(df,feature):
    ft = train[feature]
    real_ft = [x for x in ft if str(x) != 'nan']
    
    for i in range(0,len(ft)):
        if math.isnan(ft[i]):
            ft[i] = choice(real_ft)
    return ft

def grid_search(X_treino,y_treino,classifier,grid_params):
    gs = GridSearchCV(
        classifier,
        grid_params,
        verbose = 1,
        cv = 3,
        n_jobs=-1   
        )
    gs.fit(X_treino,y_treino)
    return gs.best_params_,gs. best_score_
    
def save_predictions(PassengerId_test,y_pred,classifier_name):
    results = pd.DataFrame([PassengerId_test,y_pred])
    results = results.T
    
    results.columns = ['PassengerId','Survived']
    results.to_csv(f'titanic_results_{classifier_name}_classifier.csv',index=None)    

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train.drop('PassengerId',axis=1)
y_train = train['Survived']
train = train.drop('Survived',axis=1)
train = train.drop('Name',axis=1)
train = train.drop('Ticket',axis=1)
train = train.drop('Cabin',axis=1)

PassengerId_test = test['PassengerId']
test = test.drop('PassengerId',axis=1)
test = test.drop('Name',axis=1)
test = test.drop('Cabin',axis=1)
test = test.drop('Ticket',axis=1)

train['Age'] = random_choice(train,'Age')

test['Age'] = random_choice(test,'Age')

test['Fare'] = random_choice(test,'Fare')

############################### One hot Encoding ##############################

enc = OneHotEncoder()
enc_df_embarked = pd.DataFrame(enc.fit_transform(train[['Embarked']]).toarray(),
                               columns=('Embarked_C','Embarked_Q','Embarked_S','Embarked_nan'))
enc_df_Pclass = pd.DataFrame(enc.fit_transform(train[['Pclass']]).toarray(),
                             columns=('Pclass_1','Pclass_2','Pclass_3'))
enc_df_Gender = pd.DataFrame(enc.fit_transform(train[['Sex']]).toarray(),
                             columns=('Sex_female','Sex_male'))

enc_df_embarked_test = pd.DataFrame(enc.fit_transform(test[['Embarked']]).toarray(),
                               columns=('Embarked_C','Embarked_Q','Embarked_S'))
enc_df_Pclass_test = pd.DataFrame(enc.fit_transform(test[['Pclass']]).toarray(),
                             columns=('Pclass_1','Pclass_2','Pclass_3'))
enc_df_Gender_test = pd.DataFrame(enc.fit_transform(test[['Sex']]).toarray(),
                             columns=('Sex_female','Sex_male'))


train = train.drop('Embarked',axis=1)
train[enc_df_embarked.columns] = enc_df_embarked
train = train.drop('Pclass',axis=1)
train = train.drop('Sex',axis=1)
train =pd.concat([enc_df_Gender,train],join = 'outer',axis=1)
train =pd.concat([enc_df_Pclass,train],join = 'outer',axis=1)

test = test.drop('Embarked',axis=1)
test[enc_df_embarked_test.columns] = enc_df_embarked_test
test = test.drop('Pclass',axis=1)
test = test.drop('Sex',axis=1)
test =pd.concat([enc_df_Gender_test,test],join = 'outer',axis=1)
test =pd.concat([enc_df_Pclass_test,test],join = 'outer',axis=1)

############################### One hot Encoding ##############################

############################### Normalizing ##############################

train_normalize = train[['Age','SibSp','Parch','Fare']].values
scaler = MinMaxScaler()
train_normalize = scaler.fit_transform(train_normalize)
train[['Age','SibSp','Parch','Fare']] = train_normalize
train = train.drop('Embarked_nan',axis=1)

test_normalize = test[['Age','SibSp','Parch','Fare']].values
test_normalize = scaler.transform(test_normalize)
test[['Age','SibSp','Parch','Fare']] = test_normalize

############################### Normalizing ##############################

############################### GridSearch ####################################
grid_params_knn = {
    'n_neighbors': [3,5,11,19],
    'weights': ['uniform','distance'],
    'metric': ['euclidean','manhattan']      
}

grid_params_svm = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear','poly','rbf'],
    'gamma': [0.1, 0.01,0.001, 0.0001]     
}

grid_params_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

X_treino, X_validation, y_treino, y_validation = train_test_split(train, y_train, test_size=0.10, random_state=42)

classifier = SVC()
classifier_name = 'svc'
best_params,best_score = grid_search(X_treino,y_treino,classifier,grid_params_svm)

classifier = KNeighborsClassifier()
classifier_name = 'knn'
best_params,best_score = grid_search(X_treino,y_treino,classifier,grid_params_knn)

classifier = MLPClassifier(max_iter=300)
classifier_name = 'mlp'
best_params,best_score = grid_search(X_treino,y_treino,classifier,grid_params_mlp)

print(best_params)
print(best_score)
############################### GridSearch ####################################

############################### RandomSearch ##################################
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
               n_iter = 300, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_treino, y_treino)
print(rf_random.best_params_)
print(rf_random.best_score_)
############################### RandomSearch ##################################

############################### Validation ####################################
# classifier = SVC(C= 10, gamma= 0.1, kernel = 'poly')
# classifier = KNeighborsClassifier(n_neighbors= 11, weights= 'uniform', metric = 'manhattan')
classifier = MLPClassifier(max_iter=300,activation = 'tanh', alpha= 0.05,
   hidden_layer_sizes= (50, 100, 50), learning_rate= 'constant', solver= 'adam')

classifier.fit(X_treino,y_treino)

y_pred = classifier.predict(X_validation)
classification_report(y_validation, y_pred)

clf_rf = RandomForestClassifier(n_estimators= 1400,min_samples_split= 10,
     min_samples_leaf= 2,max_features= 'sqrt', max_depth= 80, bootstrap= True)

clf_rf.fit(X_treino,y_treino)
y_pred_val = clf_rf.predict(X_validation)

classification_report(y_validation, y_pred_val)
############################### Validation ####################################

############################### Classification ################################
classifier.fit(train, y_train)
y_pred = classifier.predict(test)
save_predictions(PassengerId_test,y_pred,classifier_name)

clf_rf.fit(train, y_train)
y_pred = clf_rf.predict(test)
save_predictions(PassengerId_test,y_pred,'rf')
############################### Classification ################################





