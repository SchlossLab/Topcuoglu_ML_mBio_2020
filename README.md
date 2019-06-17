### Overview

	project
	|- README         		# the top level description of content (this doc)
	|- CONTRIBUTING    		# instructions for how to contribute to your project
	|- LICENSE         		# the license for this project
	|
	|- data/           		# raw and primary data, are not changed once created
	| |- process/     		# .tsv and .csv files generated with main.R that runs the models
	| |- baxter.0.03.subsample.shared      	# subsampled mothur generated file with OTUs from Marc Sze's analysis
	| |- metadata.tsv     		        # metadata with clinical information from Marc Sze's analysis 		
	|- code/          			# any programmatic code
	| |- learning/    			# generalization performance of model
	| |- testing/     			# building final model
	|
	|- results/        			# all output from workflows and analyses
	| |- tables/      			# tables and .Rmd code of the tables to be rendered with kable in R
	| |- figures/     			# graphs, likely designated for manuscript figures
	|
	|- submission/
	| |- manuscript.Rmd 			# executable Rmarkdown for this study, if applicable
	| |- manuscript.md 			# Markdown (GitHub) version of the *.Rmd file 
	| |- manuscript.tex 			# TeX version of *.Rmd file 
	| |- manuscript.pdf 			# PDF version of *.Rmd file 
	| |- header.tex 			# LaTeX header file to format pdf version of manuscript 
	| |- references.bib 			# BibTeX formatted references 
	|
	|- Makefile	 # Reproduce the manuscript, figures and tables



### How to regenerate this repository in R

1. Clone the Github Repository and change directory to the project directory. 

```
git clone https://github.com/BTopcuoglu/DeepLearning
cd DeepLearning
```

2. Our dependencies:

	* R version 3.5.0 
	* The R packages which needs to be installed in our environment: `caret` ,`rpart`, `xgboost`, `randomForest`, `kernlab`,`LiblineaR`, `pROC`, `tidyverse`, `cowplot`, `ggplot2`, `vegan`,`gtools`, `reshape2`. 
	* Everything needs to be run from project directory.
	* We get the OTU abundances, FIT results and Colonoscopy diagnosis from Marc's Meta study using the script ```code/learning/load_datasets.batch``` (which is included in the Makefile).
	* We update the `caret` package with my modifications using the script ```code/learning/load_caret_models.R``` . Take a look at this script to change the R packages directory where `caret` is installed. 

### How to regenerate this repository in python (in progress)


#### To run L2 Logistic Regression, L1 and L2 Linear SVM, RBF SVM, Decision Tree, Random Forest and XGBoost in Python
1. Generate tab-delimited files: Cross-validation and testing AUC scores of each model.
2. Generate tab-delimited files: The AUC scores of each hyper-parameter tested for each model.
3. Generate a comma-seperated file: The hyper-parameters tuned for each model in one file.
4. Generate ROC curve figures: The cross-validation and testing ROC curves for each model. 

```
python code/learning/main.py
```


