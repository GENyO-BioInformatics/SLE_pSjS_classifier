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

expression = pd.read_table('./data/expression.tsv')
methylation = pd.read_table('./data/methylation.tsv')
metadata = pd.read_table('./data/Metadata.tsv')

metadata["Pathology"] = metadata.Diagnosis.map({"Healthy":0, "SLE":1, "pSjS":2})

genes = pd.read_table("./Results/Biomarkers_Expr.txt", header=None)
genes = list(genes[0])
expression_biomarkers = expression.loc[genes]

CpGs = pd.read_table("./Results/Biomarkers_Meth.txt", header=None)
CpGs = list(CpGs[0])
methylation_biomarkers = methylation.loc[CpGs]

mergedData = pd.concat([expression_biomarkers, methylation_biomarkers])

bestParamsMerged = {}
test_Results_Merged = {}

metaCluster = metadata[metadata['Clusters'].isin(["Healthy", "Lymphoid", "Inflammatory", "Interferon"])]
MergedCluster = mergedData.loc[:, map(str, metaCluster.index.values.tolist())]
MergedCluster_trans = MergedCluster.transpose()

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

for i in range(10):
  
  model = xgb.XGBClassifier(objective = 'multi:softmax', tree_method = 'gpu_hist', num_class = 3, n_jobs=8)
  Merged_train, Merged_validation, MergedLabels_train, MergedLabels_validation = train_test_split(MergedCluster_trans, metaCluster.Pathology, test_size=0.20, random_state=i)
  
  # SCALING
  scaler = StandardScaler()
  Merged_train_scaled = scaler.fit_transform(Merged_train)
  Merged_validation_scaled = scaler.transform(Merged_validation)
  
  print(i)
  # define model evaluation method
  cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
  # define search
  search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100,
                              cv=cv, random_state=i, n_jobs=4, scoring=make_scorer(matthews_corrcoef))

  results = search.fit(Merged_train_scaled, MergedLabels_train)
  bestParamsMerged[i] = results.best_params_
  model = xgb.XGBClassifier(
                      eta = bestParamsMerged[i]['eta'],
                      max_depth = bestParamsMerged[i]['max_depth'], gamma = bestParamsMerged[i]['gamma'],
                      reg_alpha = bestParamsMerged[i]['alpha'], reg_lambda = bestParamsMerged[i]['lambda'],
                      subsample = bestParamsMerged[i]['subsample'],
                      min_child_weight = bestParamsMerged[i]['min_child_weight'],
                      colsample_bytree = bestParamsMerged[i]['colsample_bytree'], n_jobs=4, seed = 0,
                      objective='multi:softmax', tree_method = 'gpu_hist',
                      num_class=3)
  
  model.fit(Merged_train_scaled, MergedLabels_train)
  predicted = model.predict(Merged_validation_scaled)
  test_Results_Merged[i] = {'accuracy': metr.accuracy_score(MergedLabels_validation, predicted),
                     'precision': metr.precision_score(MergedLabels_validation, predicted, average='macro'),
                     'f1': metr.f1_score(MergedLabels_validation, predicted, average='macro'),
                     'MCC': matthews_corrcoef(MergedLabels_validation, predicted)}
  


with open('bestParamsMergedPathological', 'wb') as bestParamsMerged_file:
  pickle.dump(bestParamsMerged, bestParamsMerged_file)

with open('test_Results_Merged_Pathological', 'wb') as test_Results_Merged_file:
  pickle.dump(test_Results_Merged, test_Results_Merged_file)
  
print("Merged Results")
for metric in metrics.keys():
    allMeans = []
    for i in test_Results_Merged.keys():
        allMeans.append(test_Results_Merged[i][metric])
    meanPopulation = stats.mean(allMeans)
    sdPopulation = stats.stdev(allMeans)
    print('%s %f %f' % (metric, meanPopulation, sdPopulation))
