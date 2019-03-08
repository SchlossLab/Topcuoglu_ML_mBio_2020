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

1. Clone the Github Repository

```
git clone https://github.com/BTopcuoglu/DeepLearning
cd DeepLearning
```

2. Let's reproduce Figure 1. 

- In Figure 1 we want to plot the generalization and prediction performances of 7 CRC models.
- We are comparing classification algorithms L2-regularized logistic regression, L1 and L2-regularized support vector machines with linear kernel, radial basis function support vector machine, a decision tree, random forest and extreme gradient boosting.
- The features we use in the classification models are bacterial abundances as OTUs and fecal human hemoglobin levels.
- The labels we predict are cancer or nomal. (The patient has screen-relevant neoplasias or not.)
	- We get the OTU abundances, FIT results and Colonoscopy diagnosis from Marc's Meta study using the script ```code/learning/load_datasets.batch``` (which is included in the Makefile).
- We are expecting to generate a boxplot comparing the cross-validation and testing performances of all models.

3. Our dependencies:

* R version 3.5.0 
* The R packages which needs to be installed in our environment: "caret" ,"rpart", "xgboost", "randomForest", "kernlab","LiblineaR", "pROC", "tidyverse", "cowplot", "ggplot2", "vegan","gtools", "reshape2". 
* Everything needs to be run from project directory.

4. Let's see what we need to run to get Figure 1 using the Makefile.

```
make -n results/figures/Figure_1.pdf
```

As you see; for each model, we are running `Rscript code/learning/main.R $* "model name"` a 100 times. The argument in the middle `$*` is a seed that is set differently [0-99] for each time we run the same script. 

We provide that `$*` which is [0-99] from our array job in our HPC cluster and can submit the same script 100 times with different seed at the same time. 

 - We do this by submitting a job that runs. We finish untill this job is completed before moving on the making the figure. 

```
make data/process/best_hp_results_XGBoost_%.csv
make data/process/best_hp_results_Random_Forest_%.csv
make data/process/best_hp_results_Decision_Tree_%.csv
make data/process/best_hp_results_RBF_SVM_%.csv
make data/process/best_hp_results_L1_Linear_SVM_%.csv
make data/process/best_hp_results_L2_Linear_SVM_%.csv
```
 - Once this job is finished and 100 files are generated for each model and saved to `data/process/`. We can now run:
 
 ```
 make results/figures/Figure_1.pdf
 ```

### How to regenerate this repository in python (in progress)
```
#### To run L2 Logistic Regression, L1 and L2 Linear SVM, RBF SVM, Decision Tree, Random Forest and XGBoost in Python
1. Generate tab-delimited files: Cross-validation and testing AUC scores of each model.
2. Generate tab-delimited files: The AUC scores of each hyper-parameter tested for each model.
3. Generate a comma-seperated file: The hyper-parameters tuned for each model in one file.
4. Generate ROC curve figures: The cross-validation and testing ROC curves for each model. 

```
python code/learning/main.py
```


