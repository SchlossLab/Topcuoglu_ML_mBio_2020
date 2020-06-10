## A framework for effective application of machine learning to microbiome-based classification problems

### Abstract

Machine learning (ML) modeling of the human microbiome has the potential to identify microbial biomarkers and aid in the diagnosis of many diseases such as inflammatory bowel disease, diabetes, and colorectal cancer. Progress has been made towards developing ML models that predict health outcomes using bacterial abundances, but inconsistent adoption of training and evaluation methods call the validity of these models into question. Furthermore, there appears to be a preference by many researchers to favor increased model complexity over interpretability. To overcome these challenges, we trained seven models that used fecal 16S rRNA sequence data to predict the presence of colonic screen relevant neoplasias (SRNs; n=490 patients, 261 controls and 229 cases). We developed a reusable open-source pipeline to train, validate, and interpret ML models. To show the effect of model selection, we assessed the predictive performance, interpretability, and training time of L2-regularized logistic regression, L1 and L2-regularized support vector machines (SVM) with linear and radial basis function kernels, decision trees, random forest, and gradient boosted trees (XGBoost). The random forest model performed best at detecting SRNs with an AUROC of 0.695 [IQR 0.651-0.739] but was slow to train (83.2 h) and not inherently interpretable. Despite its simplicity, L2-regularized logistic regression followed random forest in predictive performance with an AUROC of 0.680 [IQR 0.625-0.735], trained faster (12 min), and was inherently interpretable. Our analysis highlights the importance of choosing an ML approach based on the goal of the study, as the choice will inform expectations of performance and interpretability.

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

### How to use the outlined ML pipeline for your own project 

* Please go to https://github.com/SchlossLab/ML_pipeline_microbiome.

* The current repository is to reproduce the manuscript but the provided link will take you to a user-friendly version of our pipeline. 

### How to regenerate this repository in R

Please take a look at the `Makefile` for more information about the workflow. Please also read the `submission/manuscript.pdf` to get a more detailed look on what we achieve with this ML pipeline.

1. Clone the Github Repository and change directory to the project directory.

```
git clone https://github.com/SchlossLab/Topcuoglu_ML_XXX_2019.git
cd DeepLearning
```

2. Our dependencies:

	* R version 3.5.0

	* The R packages which needs to be installed in our environment: `caret` ,`rpart`, `xgboost`, `randomForest`, `kernlab`,`LiblineaR`, `pROC`, `tidyverse`, `cowplot`, `ggplot2`, `vegan`,`gtools`, `reshape2`.

	* Everything needs to be run from project directory.

	* We need to download 2 datasets (OTU abundances and colonoscopy diagnosis of 490 patients) from *Sze MA, Schloss PD. 2018. Leveraging existing 16S rRNA gene surveys to identify reproducible biomarkers in individuals with colorectal tumors. mBio 9:e00630–18. 

	* We update the `caret` package with my modifications by running (Take a look at this script to change the R packages directory where `caret` is installed.):

		```Rscript code/learning/load_caret_models.R```

	These modifications are in `data/caret_models/svmLinear3.R` and `data/caret_models/svm_Linear4.R`

3. Follow the Makefile to generate the manuscript.


	 * The Makefile uses `code/learning/main.R` to run the pipeline which sources 4 other scripts that are part of the pipeline.

	 	* To choose the model and model hyperparemeters:`source('code/learning/model_selection.R')`

		Depending on your ML task, the model hyperparameter range to tune will be different. This is hard-coded for our study but will be updated to integrate user-defined range in the future (Issue # 10)

	 	* To preprocess and split the dataset 80-20 and to train the model: `source('code/learning/model_pipeline.R')`

	 	* To save the results of each model for each datasplit: `source('code/learning/generateAUCs.R')`

	 	* To interpret the models: `source('code/learning/permutation_importance.R')`

