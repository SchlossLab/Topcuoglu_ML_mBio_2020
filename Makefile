REFS = data/references
FIGS = results/figures
TABLES = results/tables
TEMP = data/temp
PROC = data/process
FINAL = submission/
CODE = code/learning

print-%:
	@echo '$*=$($*)'
################################################################################
#
# Part 1: Retrieve the subsampled shared file, taxonomy and metadata files that Marc Sze
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
# Part 2: Model analysis in R
#
#	Run scripts to perform all the models on the dataset and generate AUC values
#	Each model has to be submitted seperately.
#	These will generate 100 datasplit results for 7 models
#	Submit each rule on the HPC parallelized.
#	First 7 rules should finish before we move on to combining step at rule 8
#
################################################################################
$(TEMP)/traintime_XGBoost_%.csv\
$(TEMP)/all_imp_features_cor_results_XGBoost_%.csv\
$(TEMP)/all_imp_features_non_cor_results_XGBoost_%.csv\
$(TEMP)/all_hp_results_XGBoost_%.csv\
$(TEMP)/best_hp_results_XGBoost_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "XGBoost"


$(TEMP)/traintime_Random_Forest_%.csv\
$(TEMP)/all_imp_features_cor_results_Random_Forest_%.csv\
$(TEMP)/all_imp_features_non_cor_results_Random_Forest_%.csv\
$(TEMP)/all_hp_results_Random_Forest_%.csv\
$(TEMP)/best_hp_results_Random_Forest_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "Random_Forest"

$(TEMP)/traintime_Decision_Tree_%.csv\
$(TEMP)/all_imp_features_cor_results_Decision_Tree_%.csv\
$(TEMP)/all_imp_features_non_cor_results_Decision_Tree_%.csv\
$(TEMP)/all_hp_results_Decision_Tree_%.csv\
$(TEMP)/best_hp_results_Decision_Tree_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "Decision_Tree"

$(TEMP)/traintime_RBF_SVM_%.csv\
$(TEMP)/all_imp_features_cor_results_RBF_SVM_%.csv\
$(TEMP)/all_imp_features_non_cor_results_RBF_SVM_%.csv\
$(TEMP)/all_hp_results_RBF_SVM_%.csv\
$(TEMP)/best_hp_results_RBF_SVM_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "RBF_SVM"

$(TEMP)/traintime_L1_Linear_SVM_%.csv\
$(TEMP)/all_imp_features_cor_results_L1_Linear_SVM_%.csv\
$(TEMP)/all_imp_features_non_cor_results_L1_Linear_SVM_%.csv\
$(TEMP)/all_hp_results_L1_Linear_SVM_%.csv\
$(TEMP)/best_hp_results_L1_Linear_SVM_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "L1_Linear_SVM"

$(TEMP)/traintime_L2_Linear_SVM_%.csv\
$(TEMP)/all_imp_features_cor_results_L2_Linear_SVM_%.csv\
$(TEMP)/all_imp_features_non_cor_results_L2_Linear_SVM_%.csv\
$(TEMP)/all_hp_results_L2_Linear_SVM_%.csv\
$(TEMP)/best_hp_results_L2_Linear_SVM_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "L2_Linear_SVM"

$(TEMP)/traintime_L2_Logistic_Regression_%.csv\
$(TEMP)/all_imp_features_cor_results_L2_Logistic_Regression_%.csv\
$(TEMP)/all_imp_features_non_cor_results_L2_Logistic_Regression_%.csv\
$(TEMP)/all_hp_results_L2_Logistic_Regression_%.csv\
$(TEMP)/best_hp_results_L2_Logistic_Regression_%.csv	:	data/baxter.0.03.subsample.shared\
														data/metadata.tsv\
														$(CODE)/generateAUCs.R\
														$(CODE)/model_pipeline.R\
														$(CODE)/model_interpret.R\
														$(CODE)/main.R\
														$(CODE)/model_selection.R
			Rscript code/learning/main.R $* "L2_Logistic_Regression"

# Create variable names with patterns to describe temporary files

SEEDS=$(shell seq 0 99)
OBJECTS=L1_Linear_SVM L2_Linear_SVM L2_Logistic_Regression RBF_SVM Decision_Tree Random_Forest XGBoost

BEST_REPS_FILES = $(foreach S,$(SEEDS),$(foreach O,$(OBJECTS),$(TEMP)/best_hp_results_$(O)_$(S).csv))
ALL_REPS_FILES = $(foreach S,$(SEEDS),$(foreach O,$(OBJECTS),$(TEMP)/all_hp_results_$(O)_$(S).csv))
COR_IMP_REPS_FILES = $(foreach S,$(SEEDS),$(foreach O,$(OBJECTS),$(TEMP)/all_imp_features_cor_results_$(O)_$(S).csv))
NON_COR_IMP_REPS_FILES = $(foreach S,$(SEEDS),$(foreach O,$(OBJECTS),$(TEMP)/all_imp_features_non_cor_results_$(O)_$(S).csv))
TIME_REPS_FILES = $(foreach S,$(SEEDS),$(foreach O,$(OBJECTS),$(TEMP)/traintime_$(O)_$(S).csv))

# Create variable names with patterns to describe processed files that are combined

BEST_COMB_FILES = $(foreach O,$(OBJECTS),$(PROC)/combined_best_hp_results_$(O).csv)
ALL_COMB_FILES = $(foreach O,$(OBJECTS),$(PROC)/combined_all_hp_results_$(O).csv)
COR_COMB_FILES = $(foreach O,$(OBJECTS),$(PROC)/combined_all_imp_features_cor_results_$(O).csv)
NON_COR_COMB_FILES = $(foreach O,$(OBJECTS),$(PROC)/combined_all_imp_features_non_cor_results_$(O).csv)
TIME_COMB_FILES = $(foreach O,$(OBJECTS),$(PROC)/traintime_$(O).csv)

# Combine all the files generated from each submitted job

$(BEST_COMB_FILES)\
$(ALL_COMB_FILES)\
$(COR_COMB_FILES)\
$(NON_COR_COMB_FILES)\
$(TIME_COMB_FILES)\	:	$(BEST_REPS_FILES)\
						$(ALL_REPS_FILES)\
						$(COR_IMP_REPS_FILES)\
						$(NON_COR_IMP_REPS_FILES)\
						$(TIME_REPS_FILES)\
						code/cat_csv_files.sh
	bash code/cat_csv_files.sh

# Take the individual correlated importance files of linear models which have weights of each feature for each datasplit and create feature rankings for each datasplit
# Then combine each feature ranking into 1 combined file
DATA=feature_ranking

$(PROC)/combined_L1_Linear_SVM_$(DATA).tsv\
$(PROC)/combined_L2_Linear_SVM_$(DATA).tsv\
$(PROC)/combined_L2_Logistic_Regression_$(DATA).tsv	:	$(L2_LOGISTIC_REGRESSION_COR_IMP_REPS)\
												$(L1_LINEAR_SVM_COR_IMP_REPS)\
												$(L2_LINEAR_SVM_COR_IMP_REPS)\
												code/learning/get_feature_rankings.R\
												code/merge_feature_ranks.sh
	Rscript code/learning/get_feature_rankings\
	bash code/merge_feature_ranks.sh


################################################################################
#
# Part 3: Figure and table generation
#
#	Run scripts to generate figures and tables
#
################################################################################

# Figure 2 shows the generalization performance of all the models tested.
submission/Figure_2.tiff	:	$(CODE)/functions.R\
							$(CODE)/Figure2.R\
							$(BEST_COMB_FILES)
					Rscript $(CODE)/Figure2.R

# Figure 3 shows the linear model interpretation with weight rankings
submission/Figure_3.tiff	:	$(CODE)/functions.R\
							$(CODE)/Figure3.R\
							data/baxter.taxonomy\
							$(PROC)/combined_L1_Linear_SVM_$(DATA).tsv\
							$(PROC)/combined_L2_Linear_SVM_$(DATA).tsv\
							$(PROC)/combined_L2_Logistic_Regression_$(DATA).tsv
					Rscript $(CODE)/Figure3.R

# Figure 4 shows non-linear model interpretation with permutation importance
submission/Figure_4.tiff	:	$(CODE)/functions.R\
							$(CODE)/Figure4.R\
							$(BEST_COMB_FILES)\
							$(COR_COMB_FILES)\
							$(NON_COR_COMB_FILES)
					Rscript $(CODE)/Figure4.R

# Figure 5 shows training times of each model

submission/Figure_5.tiff	:	$(CODE)/functions.R\
							$(CODE)/Figure5.R\
							$(TIME_COMB_FILES)
					Rscript $(CODE)/Figure5.R

# Figure S1 shows the hyper-parameter tuning AUC values of linear models
submission/Figure_S1.tiff	:	$(CODE)/functions.R\
							$(CODE)/FigureS1.R\
							$(ALL_COMB_FILES)
					Rscript $(CODE)/FigureS1.R

# Figure S2 shows the hyper-parameter tuning AUC values of non-linear models

submission/Figure_S2.tiff	:	$(CODE)/functions.R\
							$(CODE)/FigureS1.R\
							$(ALL_COMB_FILES)
					Rscript $(CODE)/FigureS2.R


# Table 1 is a summary of the compelxity properties of all the models tested.
submission/TableS1.pdf :	submission/Table1.Rmd\
						submission/header.tex
	R -e "rmarkdown::render('submission/Table1.Rmd', clean=TRUE)"


################################################################################
#
# Part 4: Pull it all together
#
# Render the manuscript
#
################################################################################


submission/manuscript.%	:	submission/mbio.csl\
							submission/references.bib\
							submission/manuscript.Rmd
	R -e 'rmarkdown::render("submission/manuscript.Rmd", clean=FALSE)'
	mv submission/manuscript.knit.md submission/manuscript.md
	rm submission/manuscript.utf8.md


write.paper :	submission/Figure_1.tiff\
				submission/Figure_2.tiff\
				submission/Figure_3.tiff\
				submission/Figure_4.tiff\
				submission/Figure_5.tiff\
				submission/Figure_S1.tiff\
				submission/Figure_S1.tiff\
				submission/manuscript.Rmd\
				submission/manuscript.md\
				submission/manuscript.tex\
				submission/manuscript.pdf


# module load perl-modules latexdiff/1.2.0
submission/marked_up.pdf : submission/manuscript.tex
	git cat-file -p b7118145861ded9:submission/manuscript.tex > submission/manuscript_old.tex
	latexdiff submission/manuscript_old.tex submission/manuscript.tex > submission/marked_up.tex
	pdflatex -output-directory=submission submission/marked_up.tex
	rm submission/marked_up.aux
	rm submission/marked_up.log
	rm submission/marked_up.out
	rm submission/marked_up.tex
	rm submission/manuscript_old.tex

submission/manuscript.docx : submission/manuscript.tex
	pandoc submission/manuscript.tex -o submission/manuscript.docx
