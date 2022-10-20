import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
import sklearn.metrics as metr
import xgboost as xgb
import statistics as stats
import pickle



####################
### DATA LOADING ###
####################
expression = pd.read_table('./data/expression.tsv')
methylation = pd.read_table('./data/methylation.tsv')
metadata = pd.read_table('./data/Metadata.tsv')

metadata["Pathology"] = metadata.Diagnosis.map({"Healthy":0, "SLE":1, "pSjS":2})

# We need samples in columns
expression_trans = expression.transpose() 
methylation_trans =  methylation.transpose()


##################################
### HYPERPARAMETERS OPTIMIZATION ###
##################################

# Create the random grid
random_grid = {'eta': [0.001, 0.01, 0.1],
               'gamma': [0, 0.1, 0.3],
               'max_depth': [1, 3, 5, 7, 9],
               'min_child_weight': [int(x) for x in np.linspace(1, 50, num = 10)],
               'subsample': [0.7, 0.8, 0.9],
               'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
               'alpha': [0, 0.1, 0.3],
               'lambda': [1, 2, 3]
               }

metrics = {"accuracy":make_scorer(metr.accuracy_score), "precision":make_scorer(metr.precision_score, average='micro'),
           "f1":make_scorer(metr.f1_score, average='micro'),
           "MCC":make_scorer(matthews_corrcoef)}
           
# Expression
bestParamsExpr = {}
test_Results_Expr = {}

for i in range(10):
  model = xgb.XGBClassifier(objective = 'multi:softmax', tree_method = 'gpu_hist', num_class = 3, n_jobs=4)
  expr_train, expr_validation, exprLabels_train, exprLabels_validation = train_test_split(expression_trans, metadata.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  expr_train_scaled = scaler.fit_transform(expr_train)
  expr_validation_scaled = scaler.transform(expr_validation)
  
  cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
  search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100,
                              cv=cv, random_state=i, n_jobs=6, scoring=make_scorer(matthews_corrcoef))

  results = search.fit(expr_train_scaled, exprLabels_train)
  bestParamsExpr[i] = results.best_params_
  model = xgb.XGBClassifier(
                      eta = bestParamsExpr[i]['eta'],
                      max_depth = bestParamsExpr[i]['max_depth'], gamma = bestParamsExpr[i]['gamma'],
                      reg_alpha = bestParamsExpr[i]['alpha'], reg_lambda = bestParamsExpr[i]['lambda'],
                      subsample = bestParamsExpr[i]['subsample'],
                      min_child_weight = bestParamsExpr[i]['min_child_weight'],
                      colsample_bytree = bestParamsExpr[i]['colsample_bytree'], n_jobs=4, seed = 0,
                      objective='multi:softmax', tree_method = 'gpu_hist',
                      num_class=3)
  
  model.fit(expr_train_scaled, exprLabels_train)
  predicted = model.predict(expr_validation_scaled)
  test_Results_Expr[i] = {'accuracy': metr.accuracy_score(exprLabels_validation, predicted),
                     'precision': metr.precision_score(exprLabels_validation, predicted, average='macro'),
                     'f1': metr.f1_score(exprLabels_validation, predicted, average='macro'),
                     'MCC': matthews_corrcoef(exprLabels_validation, predicted)}
  


with open('bestParamsExpr', 'wb') as bestParamsExpr_file:
  pickle.dump(bestParamsExpr, bestParamsExpr_file)

with open('test_Results_Expr', 'wb') as test_Results_Expr_file:
  pickle.dump(test_Results_Expr, test_Results_Expr_file)
  
print("Test Results")
for metric in metrics.keys():
    allMeans = []
    for i in test_Results_Expr.keys():
        allMeans.append(test_Results_Expr[i][metric])
    meanPopulation = stats.mean(allMeans)
    sdPopulation = stats.stdev(allMeans)
    print('%s %f %f' % (metric, meanPopulation, sdPopulation))


# Methylation
bestParamsMeth = {}
test_Results_Meth = {}

for i in range(1,10):
  
  model = xgb.XGBClassifier(objective = 'multi:softmax', tree_method = 'gpu_hist', num_class = 3, n_jobs=8)
  meth_train, meth_validation, methLabels_train, methLabels_validation = train_test_split(methylation_trans, metadata.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  meth_train_scaled = scaler.fit_transform(meth_train)
  meth_validation_scaled = scaler.transform(meth_validation)
  

  cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
  search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100,
                              cv=cv, random_state=i, n_jobs=4, scoring=make_scorer(matthews_corrcoef))

  results = search.fit(meth_train_scaled, methLabels_train)
  bestParamsMeth[i] = results.best_params_
  model = xgb.XGBClassifier(
                      eta = bestParamsMeth[i]['eta'],
                      max_depth = bestParamsMeth[i]['max_depth'], gamma = bestParamsMeth[i]['gamma'],
                      reg_alpha = bestParamsMeth[i]['alpha'], reg_lambda = bestParamsMeth[i]['lambda'],
                      subsample = bestParamsMeth[i]['subsample'],
                      min_child_weight = bestParamsMeth[i]['min_child_weight'],
                      colsample_bytree = bestParamsMeth[i]['colsample_bytree'], n_jobs=4, seed = 0,
                      objective='multi:softmax', tree_method = 'gpu_hist',
                      num_class=3)
  
  model.fit(meth_train_scaled, methLabels_train)
  predicted = model.predict(meth_validation_scaled)
  test_Results_Meth[i] = {'accuracy': metr.accuracy_score(methLabels_validation, predicted),
                     'precision': metr.precision_score(methLabels_validation, predicted, average='macro'),
                     'f1': metr.f1_score(methLabels_validation, predicted, average='macro'),
                     'MCC': matthews_corrcoef(methLabels_validation, predicted)}
  


with open('bestParamsMeth', 'wb') as bestParamsMeth_file:
  pickle.dump(bestParamsMeth, bestParamsMeth_file)

with open('test_Results_Meth', 'wb') as test_Results_Meth_file:
  pickle.dump(test_Results_Meth, test_Results_Meth_file)
  
print("Test Results")
for metric in metrics.keys():
    allMeans = []
    for i in test_Results_Meth.keys():
        allMeans.append(test_Results_Meth[i][metric])
    meanPopulation = stats.mean(allMeans)
    sdPopulation = stats.stdev(allMeans)
    print('%s %f %f' % (metric, meanPopulation, sdPopulation))

