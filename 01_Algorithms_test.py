import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from lazypredict.Supervised import LazyClassifier
from functools import reduce


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


###################
### LAZYPREDICT ###
###################
lazyResultsExpr = []
lazyResultsMeth = []
for iteration in range(10):
  expr_train, expr_validation, exprLabels_train, exprLabels_validation = train_test_split(expression_trans, metadata.Pathology, test_size=0.20, random_state=iteration)
  meth_train, meth_validation, methLabels_train, methLabels_validation = train_test_split(methylation_trans, metadata.Pathology, test_size=0.20, random_state=iteration)

  clf = LazyClassifier(custom_metric=matthews_corrcoef)
  models, predictions = clf.fit(expr_train, expr_validation, exprLabels_train, exprLabels_validation)
  lazyResultsExpr.append(models.loc[:, "matthews_corrcoef"])
  
  clf = LazyClassifier(custom_metric=matthews_corrcoef)
  models, predictions = clf.fit(meth_train, meth_validation, methLabels_train, methLabels_validation)
  lazyResultsMeth.append(models.loc[:, "matthews_corrcoef"])


lazyResultsExprMerged = reduce(lambda left,right: pd.merge(left, right, on="Model"), lazyResultsExpr)
lazyResultsExprMerged.to_csv("lazypredict_expr.tsv", sep = "\t")

lazyResultsMethMerged = reduce(lambda left,right: pd.merge(left, right, on="Model"), lazyResultsMeth)
lazyResultsMethMerged.to_csv("lazypredict_meth.tsv", sep = "\t")
