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
* Python 3.6.5
* Latest PyTorch 
* Latest Sklearn
#### Run the python code you choose
```
git clone https://github.com/BTopcuoglu/DeepLearning
```
