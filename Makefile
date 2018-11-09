REFS = data/references
FIGS = results/figures
TABLES = results/tables
PROC = data/process
FINAL = submission/

# utility function to print various variables. For example, running the
# following at the command line:
#
#	make print-BAM
#
# will generate:
#	BAM=data/raw_june/V1V3_0001.bam data/raw_june/V1V3_0002.bam ...
print-%:
	@echo '$*=$($*)'

################################################################################
#
# Part 1: Retrieve the subsampled shared file and metadata files that Marc Sze
# published in https://github.com/SchlossLab/Sze_CRCMetaAnalysis_mBio_2018
#
#	Copy from Github
#
################################################################################

data/baxter.0.03.subsample.shared\
data/metadata.tsv	:	code/learning/load_datasets.batch
	bash code/learning/load_datasets.batch

################################################################################
#
# Part 2: Model analysis in python
#
#	Run scripts to perform all the models on the dataset and generate AUC values
#
################################################################################


$(PROC)/L2_Logistic_Regression.tsv\
$(PROC)/L1_SVM_Linear_Kernel.tsv\
$(PROC)/SVM_RBF.tsv\
$(PROC)/L2_SVM_Linear_Kernel.tsv\
$(PROC)/Random_Forest.tsv\
$(PROC)/Decision_Tree.tsv\
$(PROC)/XGBoost.tsv	:	data/baxter.0.03.subsample.shared\
						data/metadata.tsv\
						code/learning/main.py\
						code/learning/preprocess_data.py\
						code/learning/model_selection.py
	python3 code/learning/main.py



################################################################################
#
# Part 3: Figure and table generation
#
#	Run scripts to generate figures and tables
#
################################################################################

$(FIGS)/Figure1.pdf :	$(PROC)/L2_Logistic_Regression.tsv\
						$(PROC)/L1_SVM_Linear_Kernel.tsv\
						$(PROC)/SVM_RBF.tsv\
						$(PROC)/L2_SVM_Linear_Kernel.tsv\
						$(PROC)/Random_Forest.tsv\
						$(PROC)/Decision_Tree.tsv\
						$(PROC)/XGBoost.tsv
						code/learning/compareAUC.R
	R -e "source('code/learning/compareAUC.R')"

$(TABLES)/Table1.pdf :	$(PROC)/model_parameters.txt\
						$(TABLES)/Table1.Rmd\
						$(TABLES)/header.tex
	R -e "rmarkdown::render('$(TABLES)/Table1.Rmd', clean=TRUE)"
	rm $(TABLES)/Table1.tex

$(TABLES)/Table2.pdf :	$(PROC)/param_grid.csv\
						$(TABLES)/Table2.Rmd\
						$(TABLES)/header.tex
	R -e "rmarkdown::render('$(TABLES)/Table2.Rmd', clean=TRUE)"
	rm $(TABLES)/Table2.tex

$(TABLES)/Table3.pdf :	$(PROC)/param_grid.csv\
						$(TABLES)/Table3.Rmd\
						$(TABLES)/header.tex
	R -e "rmarkdown::render('$(TABLES)/Table3.Rmd', clean=TRUE)"
	rm $(TABLES)/Table3.tex









################################################################################
#
# Part 4: Pull it all together
#
# Render the manuscript
#
################################################################################


$(FINAL)/manuscript.% : 			\ #include data files that are needed for paper don't leave this line with a : \
						$(FINAL)/mbio.csl\
						$(FINAL)/references.bib\
						$(FINAL)/manuscript.Rmd
	R -e 'render("$(FINAL)/manuscript.Rmd", clean=FALSE)'
	mv $(FINAL)/manuscript.knit.md submission/manuscript.md
	rm $(FINAL)/manuscript.utf8.md


write.paper : $(TABLES)/table_1.pdf $(TABLES)/table_2.pdf\ #customize to include
				$(FIGS)/figure_1.pdf $(FIGS)/figure_2.pdf\	# appropriate tables and
				$(FIGS)/figure_3.pdf $(FIGS)/figure_4.pdf\	# figures
				$(FINAL)/manuscript.Rmd $(FINAL)/manuscript.md\
				$(FINAL)/manuscript.tex $(FINAL)/manuscript.pdf
