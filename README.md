# Multiclass classifier for differential diagnosis of SLE, pSjS and healthy controls
This is the code associated to the article "Machine Learning for differential diagnosis of Systemic Lupus Erythematosus and Sjögren’s Syndrome based on gene expression and DNA methylation". The biomarkers validation with external data may be reproduced using this code.

On the first place, run the R script prepare_external_data.R to download and process the data from the GEO repository. There are some dependences that should be installed with the following code in an R session:
```
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("biomaRt", "GEOquery", "minfi", "lumi"))
```
Then, run the script
