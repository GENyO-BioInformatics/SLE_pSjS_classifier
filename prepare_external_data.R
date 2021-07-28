##############################
### GSE108497 (Expression) ###
##############################

# Decompress and load the expression matrix
unzip("Validation_data/GSE108497.zip", exdir = "Validation_data/") # Data downloaded from https://adex.genyo.es
expr_GSE108497 <- read.delim("Validation_data/GSE108497.tsv", row.names = 1)

# Transform gene symbols to ENSEMBL IDs
library(biomaRt)
ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl", version = 75, GRCh = 37)

GSE108497_ENSEMBL <- getBM(attributes = c('ensembl_gene_id', "external_gene_name"),
						   filters = 'external_gene_name',
						   values = rownames(expr_GSE108497),
						   mart = ensembl)
rownames(GSE108497_ENSEMBL) <- GSE108497_ENSEMBL[,1]

# Select the biomarkers available
biomarkersExpr <- read.delim("Validation_data/biomarkers_expression.txt", header = F)
GSE108497_ENSEMBL <- GSE108497_ENSEMBL[biomarkersExpr[,1],]
expr_GSE108497 <- expr_GSE108497[rownames(expr_GSE108497) %in% GSE108497_ENSEMBL$external_gene_name,]

# Save the expression table
write.table(expr_GSE108497, "Validation_data/GSE108497_expr.tsv", sep="\t", quote = F)


###############################
### GSE166373 (Methylation) ###
###############################

# Download the data from GEO 
library(GEOquery)
getGEOSuppFiles("GSE166373")
untar("GSE166373/GSE166373_RAW.tar", exdir = "IDAT")
GEOData <- getGEO("GSE166373")
pheno <- pData(phenoData(GEOData[[1]]))

# Read the IDAT files
library(minfi)
basenames_split <- strsplit(pheno$supplementary_file, "suppl/")
basenames <- sapply(basenames_split, function(x) {return(x[2])})
targets <- data.frame("Sample_Name" = pheno$title, "Basename" = basenames)

rgchannelset <- read.metharray.exp(targets = targets, base = "IDAT")
sampleNames(rgchannelset) <- targets$Sample_Name

# Perform Noob normalization
mset <- preprocessNoob(rgchannelset)
rset <- ratioConvert(mset)
grset <- mapToGenome(rset)

# Select the biomarkers from the methylation matrix
beta.noob <- getBeta(grset)
beta.noob <- beta.noob[biomarkersMeth[,1],]

# Repeat for the samples measured with the EPIC platform
pheno2 <- pData(phenoData(GEOData[[2]]))
basenames_split <- strsplit(pheno2$supplementary_file, "suppl/")
basenames <- sapply(basenames_split, function(x) {return(x[2])})
targets <- data.frame("Sample_Name" = pheno2$title, "Basename" = basenames)

rgchannelset <- read.metharray.exp(targets = targets, base = "IDAT")
sampleNames(rgchannelset) <- targets$Sample_Name

mset <- preprocessNoob(rgchannelset)
rset <- ratioConvert(mset)
grset <- mapToGenome(rset)

beta.noob2 <- getBeta(grset)
beta.noob2 <- beta.noob2[biomarkersMeth[,1],]

# Combine the samples and transform beta-values to M-values
betaGSE166373 <- cbind(beta.noob, beta.noob2)
mvalGSE166373 <- lumi::beta2m(betaGSE166373)

# Prepare the metadata
metaGSE166373 <- rbind(pheno, pheno2)
metaGSE166373 <- metaGSE166373[,c("geo_accession","description")]

colnames(mvalGSE166373) <- rownames(metaGSE166373)

# Save the files
write.table(mvalGSE166373, "Validation_data/GSE166373_meth.tsv", sep="\t", quote = F)
write.table(metaGSE166373, "Validation_data/GSE166373_meta.tsv", sep="\t", quote = F)
