### Overview

	project
	|- README          # the top level description of content (this doc)
	|- CONTRIBUTING    # instructions for how to contribute to your project
	|- LICENSE         # the license for this project
	|
	|- data           # raw and primary data, are not changed once created
	|
	|- code/          # any programmatic code
	| |- learning     # generalization performance of model
	| |- testing      # building final model
	|
	|- results        # all output from workflows and analyses
	| |- tables/      # text version of tables to be rendered with kable in R
	| |- figures/     # graphs, likely designated for manuscript figures
	|
	|- submission/
	| |- study.Rmd # executable Rmarkdown for this study, if
	| |applicable - study.md # Markdown (GitHub) version of the
	| |*.Rmd file - study.tex # TeX version of *.Rmd file -
	| |study.pdf # PDF version of *.Rmd file - header.tex # LaTeX
	| |header file to format pdf version of manuscript -
	| |references.bib # BibTeX formatted references - XXXX.csl # csl
	| |file to format references for journal XXX



### How to regenerate this repository

#### Dependencies and locations
* Python 3.6.5 or Python 2.7, Matplotlib, Numpy, Scipy, Sympy, Pandas, Sklearn and XGBoost to run Shallow Learning code. 
* If running Deep Learning code you need to have Python 3 and Latest PyTorch and Latest Keras with Theano backend.
* Run everything from project directory.

#### Run the following code
```
git clone https://github.com/BTopcuoglu/DeepLearning
```
#### To run L2 Logistic Regression, L1 and L2 Linear SVM, RBF SVM, Decision Tree, Random Forest and XGBoost
1. Generate tab-delimited files: Cross-validation and testing AUC scores of each model.
2. Generate tab-delimited files: The AUC scores of each hyper-parameter tested for each model.
3. Generate a comma-seperated file: The hyper-parameters tuned for each model in one file.
4. Generate ROC curve figures: The cross-validation and testing ROC curves for each model. 

```
python code/learning/main.py
```
#### The Makefile will reproduce all the other figures and tables used in the manuscript.
```
make submission/manuscript.pdf
```


