# Multiclass classifier for differential diagnosis of SLE, pSjS and healthy controls
This is the code associated to the article "Differential diagnosis of Systemic Lupus Erythematosus and Sjogren’s Syndrome using machine learning and multi-omics data".

## 01. Test algorithms
With the code **01_Algorithms_test.py**, 29 different machine learning models are tested with a expression and a methylation matrices. The process is repeated 10 times with different training/test splits. The obtained Matthew's correlation coefficient (MCC) for each iteration are written to **lazypredict_expr.tsv** and **lazypredict_meth.tsv**.

## 02. Overall models
Use **02_Overall_models.py** to train and test a XGBoost model in 10 iterations. The optimized hyperparameters for each iteration are saved in **bestParamsExpr** and **bestParamsMeth** files. The classification metrics in train tests for each iteration are saved in **test_Results_Expr** and **test_Results_Meth** files and are printed to screen.

## 03. Biomarkers selection
**03_Biomarkers.py** can be used to select subsets of genes and CpGs for gene expression and methylation repectively using the hyperparameters learnt in the previous optimization fir each iteration. The selected features are written to **Results/Biomarkers_Expr.txt** and **Results/Biomarkers_Meth.txt** and the mean MCC after feature selection are printed to screen.

## 04. Integration
With **04_Integration.py**, the selected features from the previous script are integrated into a new matrix combining gene expression and methylation data. This integrated matrix is used to construct new XGBoost models and to test their performance.

## 05. Prepare external data
Use the script **05_prepare_external_data.R** to download and process the validation data from the GEO repository. There are some dependences that should be installed with the following code in an R session:
```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("biomaRt", "GEOquery", "minfi", "lumi"))
```

The files **GSE108497_expr.tsv**, **GSE166373_meth.tsv** and **GSE166373_meta.tsv** will be written in the Validation_data folder.

## 06. Validation
With the code **06_Validation.py**, the public expression and methylation data are analyzed with XGBoost and the selected biomarkers.
