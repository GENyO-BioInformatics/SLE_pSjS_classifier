# Load the necessary modules and functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Load data and metadata
methylationGSE166373 = pd.read_table('Validation_data/GSE166373_meth.tsv')
metadataGSE166373 = pd.read_table('Validation_data/GSE166373_meta.tsv')

metadataGSE166373["Pathology"] = metadataGSE166373.Condition.map({"Healthy":0, "SjS":1})

# Normalization
methylationGSE166373_trans = methylationGSE166373.transpose()
methylationGSE166373_transscaled = StandardScaler().fit_transform(methylationGSE166373_trans)



# Define possible values for the RF models

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
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# define model
model = RandomForestClassifier()

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
import sklearn.metrics as metr
import statistics as stats

metrics = {"accuracy":make_scorer(metr.accuracy_score),
           "MCC":make_scorer(matthews_corrcoef)}

bestParams = {}
test_Results = {}

for i in range(1, 21):
    print(i)
    # define model evaluation method
    cv = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
    # define search
    search = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100,
                                cv=cv, random_state=i, n_jobs=24)
    meth_train, meth_validation, methLabels_train, methLabels_validation = train_test_split(methylationGSE166373_transscaled,
                                                                                            metadataGSE166373.Pathology,
                                                                                            test_size=0.20,
                                                                                            random_state=i)
    results = search.fit(meth_train, methLabels_train)
    bestParams[i] = results.best_params_

    n_estimators = bestParams[i]['n_estimators']
    max_features = bestParams[i]['max_features']
    max_depth = bestParams[i]['max_depth']
    min_samples_split = bestParams[i]['min_samples_split']
    min_samples_leaf = bestParams[i]['min_samples_leaf']
    bootstrap = bestParams[i]['bootstrap']

    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   max_features=max_features, max_depth=max_depth, bootstrap=bootstrap,
                                   random_state=0)

    model.fit(meth_train, methLabels_train)
    predicted = model.predict(meth_validation)
    test_Results[i] = {'accuracy': metr.accuracy_score(methLabels_validation, predicted),
                       'MCC': matthews_corrcoef(methLabels_validation, predicted)}


print("RF Test Results")
for metric in metrics.keys():
    allMeans = []
    for i in test_Results.keys():
        allMeans.append(test_Results[i][metric])

    meanPopulation = stats.mean(allMeans)
    sdPopulation = stats.stdev(allMeans)
    se = sdPopulation
    z = 1.96 # For 95 % CI
    lcb = meanPopulation - z * se  # lower limit of the CI
    ucb = meanPopulation + z * se  # upper limit of the CI

    print('%s: %f %f %f %f' % (metric, meanPopulation, sdPopulation, lcb, ucb))
