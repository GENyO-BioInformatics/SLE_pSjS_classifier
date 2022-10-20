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
import pickle
import statistics

####################
### DATA LOADING ###
####################
expression = pd.read_table('./data/expression.tsv')
methylation = pd.read_table('./data/methylation.tsv')
metadata = pd.read_table('./data/Metadata.tsv')

metadata["Pathology"] = metadata.Diagnosis.map({"Healthy":0, "SLE":1, "pSjS":2})

##################
### EXPRESSION ###
##################

# Load the hyperparameters saved with the script 02_Overall_models.py
with open('./Results/bestParamsExprPathological', 'rb') as bestParamsExprPathologicalFile:
  bestParamsExpr = pickle.load(bestParamsExprPathologicalFile)
  
# Optional: Select a subset of samples
metaCluster = metadata[metadata['Clusters'].isin(["Healthy", "Lymphoid", "Inflammatory", "Interferon"])]
exprCluster = expression.loc[:, map(str, metaCluster.index.values.tolist())]
exprCluster_trans = exprCluster.transpose()

biomarkersLists = []
MCCList = []
positionsFolds = {}

for i in range(10):
  
  expr_train, expr_validation, exprLabels_train, exprLabels_validation = train_test_split(exprCluster_trans, metaCluster.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  expr_train_scaled = scaler.fit_transform(expr_train)
  expr_validation_scaled = scaler.transform(expr_validation)
  
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
  
  weights = model.feature_importances_.tolist()
  weights = [abs(ele) for ele in weights]
  weights = dict(zip(expression.index, weights))
  weights = sorted(weights, key=weights.get, reverse=True)

  positions = dict(zip(weights, range(1, len(weights)+1)))
  positionsFolds[i] = positions
  

meanPositions = {}
for gene in positionsFolds[1]:
    positionsGene = 0
    for i in positionsFolds:
        positionsGene += positionsFolds[i][gene]
    meanPosition = positionsGene / len(positionsFolds)
    meanPositions[gene] = meanPosition

meanPositions = sorted(meanPositions, key=meanPositions.get, reverse=False)

MCCSubsets = {}
for i in range(10):
  expr_train, expr_validation, exprLabels_train, exprLabels_validation = train_test_split(exprCluster_trans, metaCluster.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  expr_train_scaled = scaler.fit_transform(expr_train)
  expr_validation_scaled = scaler.transform(expr_validation)
  
  for nGenes in range(10, 1010, 10):
    genes = meanPositions[:nGenes]
    expr_train2 = pd.DataFrame(data=expr_train_scaled, index=exprLabels_train.index, columns=expression.index)
    expr_train2 = expr_train2[genes]
    expr_validation2 = pd.DataFrame(data=expr_validation_scaled, index=exprLabels_validation.index, columns=expression.index)
    expr_validation2 = expr_validation2[genes]

    model2 = xgb.XGBClassifier(
                    eta = bestParamsExpr[i]['eta'],
                    max_depth = bestParamsExpr[i]['max_depth'], gamma = bestParamsExpr[i]['gamma'],
                    reg_alpha = bestParamsExpr[i]['alpha'], reg_lambda = bestParamsExpr[i]['lambda'],
                    subsample = bestParamsExpr[i]['subsample'],
                    min_child_weight = bestParamsExpr[i]['min_child_weight'],
                    colsample_bytree = bestParamsExpr[i]['colsample_bytree'], n_jobs=4, seed = 0,
                    objective='multi:softmax', tree_method = 'gpu_hist',
                    num_class=3)

    model2.fit(expr_train2, exprLabels_train)
    predicted = model2.predict(expr_validation2)
    
    if i == 0:
      MCCSubsets[nGenes] = [matthews_corrcoef(exprLabels_validation, predicted)]
    else:
      MCCSubsets[nGenes].append(matthews_corrcoef(exprLabels_validation, predicted))



MCCSubsetsMeans = {}
MCCSubsetsSDs = {}

for nGenes in MCCSubsets:
    MCCSubsetsMeans[nGenes] = sum(MCCSubsets[nGenes]) / len(MCCSubsets[nGenes])
    MCCSubsetsSDs[nGenes] = statistics.stdev(MCCSubsets[nGenes])

import operator
maxnGenes = max(MCCSubsetsMeans.items(), key=operator.itemgetter(1))[0]
print("Max nGenes: " + str(maxnGenes))
print(max(MCCSubsetsMeans.values()))
print(MCCSubsetsSDs[maxnGenes])

finalGenes = meanPositions[:maxnGenes]

with open("./Results/Biomarkers_Expr.txt", 'w') as file:
  file.write('\n'.join(finalGenes))
  

##################
### Methylation ###
##################
with open('./Results/bestParamsMethPathological', 'rb') as bestParamsMethPathologicalFile:
  bestParamsMeth = pickle.load(bestParamsMethPathologicalFile)
  
metaCluster = metadata[metadata['Clusters'].isin(["Healthy", "Lymphoid", "Inflammatory", "Interferon"])]
MethCluster = methylation.loc[:, map(str, metaCluster.index.values.tolist())]
MethCluster_trans = MethCluster.transpose()

biomarkersLists = []
MCCList = []
positionsFolds = {}

for i in range(10):
  
  Meth_train, Meth_validation, MethLabels_train, MethLabels_validation = train_test_split(MethCluster_trans, metaCluster.Pathology, test_size=0.20, random_state=i)
 
  # SCALING
  scaler = StandardScaler()
  Meth_train_scaled = scaler.fit_transform(Meth_train)
  Meth_validation_scaled = scaler.transform(Meth_validation)
  
  model = xgb.XGBClassifier(
                      eta = bestParamsMeth[i]['eta'],
                      max_depth = bestParamsMeth[i]['max_depth'], gamma = bestParamsMeth[i]['gamma'],
                      reg_alpha = bestParamsMeth[i]['alpha'], reg_lambda = bestParamsMeth[i]['lambda'],
                      subsample = bestParamsMeth[i]['subsample'],
                      min_child_weight = bestParamsMeth[i]['min_child_weight'],
                      colsample_bytree = bestParamsMeth[i]['colsample_bytree'], n_jobs=4, seed = 0,
                      objective='multi:softmax', tree_method = 'gpu_hist',
                      num_class=3)
  
  model.fit(Meth_train_scaled, MethLabels_train)
  
  weights = model.feature_importances_.tolist()
  weights = [abs(ele) for ele in weights]
  weights = dict(zip(methylation.index, weights))
  weights = sorted(weights, key=weights.get, reverse=True)

  positions = dict(zip(weights, range(1, len(weights)+1)))
  positionsFolds[i] = positions
  

meanPositions = {}
for gene in positionsFolds[1]:
    positionsGene = 0
    for i in positionsFolds:
        positionsGene += positionsFolds[i][gene]
    meanPosition = positionsGene / len(positionsFolds)
    meanPositions[gene] = meanPosition

meanPositions = sorted(meanPositions, key=meanPositions.get, reverse=False)

MCCSubsets = {}
for i in range(10):
  Meth_train, Meth_validation, MethLabels_train, MethLabels_validation = train_test_split(MethCluster_trans, metaCluster.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  Meth_train_scaled = scaler.fit_transform(Meth_train)
  Meth_validation_scaled = scaler.transform(Meth_validation)
  
  for nGenes in range(10, 1010, 10):
    genes = meanPositions[:nGenes]
    Meth_train2 = pd.DataFrame(data=Meth_train_scaled, index=MethLabels_train.index, columns=methylation.index)
    Meth_train2 = Meth_train2[genes]
    Meth_validation2 = pd.DataFrame(data=Meth_validation_scaled, index=MethLabels_validation.index, columns=methylation.index)
    Meth_validation2 = Meth_validation2[genes]

    model2 = xgb.XGBClassifier(
                    eta = bestParamsMeth[i]['eta'],
                    max_depth = bestParamsMeth[i]['max_depth'], gamma = bestParamsMeth[i]['gamma'],
                    reg_alpha = bestParamsMeth[i]['alpha'], reg_lambda = bestParamsMeth[i]['lambda'],
                    subsample = bestParamsMeth[i]['subsample'],
                    min_child_weight = bestParamsMeth[i]['min_child_weight'],
                    colsample_bytree = bestParamsMeth[i]['colsample_bytree'], n_jobs=4, seed = 0,
                    objective='multi:softmax', tree_method = 'gpu_hist',
                    num_class=3)

    model2.fit(Meth_train2, MethLabels_train)
    predicted = model2.predict(Meth_validation2)
    
    if i == 0:
      MCCSubsets[nGenes] = [matthews_corrcoef(MethLabels_validation, predicted)]
    else:
      MCCSubsets[nGenes].append(matthews_corrcoef(MethLabels_validation, predicted))



MCCSubsetsMeans = {}
MCCSubsetsSDs = {}

for nGenes in MCCSubsets:
    MCCSubsetsMeans[nGenes] = sum(MCCSubsets[nGenes]) / len(MCCSubsets[nGenes])
    MCCSubsetsSDs[nGenes] = statistics.stdev(MCCSubsets[nGenes])

import operator
maxnGenes = max(MCCSubsetsMeans.items(), key=operator.itemgetter(1))[0]
print("Max nGenes: " + str(maxnGenes))
print(max(MCCSubsetsMeans.values()))
print(MCCSubsetsSDs[maxnGenes])

finalGenes = meanPositions[:maxnGenes]

with open("./Results/Biomarkers_Meth.txt", 'w') as file:
  file.write('\n'.join(finalGenes))
