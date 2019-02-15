REFS = data/references
FIGS = results/figures
TABLES = results/tables
PROC = data/process
FINAL = submission/
CODE = code/learning


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
# Part 2: Model analysis in R
#
#	Run scripts to perform all the models on the dataset and generate AUC values
#
################################################################################

# Define number of the array that .pbf file runs
OUT_NO=$(shell seq 0 99)

######### L1 Linear SVM files .pbs files generate #################
L1_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_L1_Linear_SVM_,$(OUT_NO))
L1_IMP_FILE=$(addsuffix .csv,$(L1_IMP_OUT_FILE))

L1_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_L1_Linear_SVM_,$(OUT_NO))
L1_BEST_FILE=$(addsuffix .csv,$(L1_BEST_OUT_FILE))

L1_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_L1_Linear_SVM_,$(OUT_NO))
L1_ALL_FILE=$(addsuffix .csv,$(L1_ALL_OUT_FILE))
###################################################################


############# L2 SVM files .pbs files generate ####################
L2_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_L2_Linear_SVM_,$(OUT_NO))
L2_IMP=$(addsuffix .csv,$(L2_IMP_OUT_FILE))

L2_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_L2_Linear_SVM_,$(OUT_NO))
L2_BEST=$(addsuffix .csv,$(L2_BEST_OUT_FILE))

L2_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_L2_Linear_SVM_,$(OUT_NO))
L2_ALL_FILE=$(addsuffix .csv,$(L2_ALL_OUT_FILE))
###################################################################


###### L2 Logistic Regression files .pbs files generate ###########
Logit_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_L2_Logistic_Regression_,$(OUT_NO))
Logit_IMP=$(addsuffix .csv,$(Logit_IMP_OUT_FILE))

Logit_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_L2_Logistic_Regression_,$(OUT_NO))
Logit_BEST=$(addsuffix .csv,$(Logit_BEST_OUT_FILE))

Logit_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_L2_Logistic_Regression_,$(OUT_NO))
Logit_ALL_FILE=$(addsuffix .csv,$(Logit_ALL_OUT_FILE))
###################################################################


######### RBF SVM files .pbs files generate #############
RBF_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_RBF_SVM_,$(OUT_NO))
RBF_IMP=$(addsuffix .csv,$(RBF_IMP_OUT_FILE))

RBF_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_RBF_SVM_,$(OUT_NO))
RBF_BEST=$(addsuffix .csv,$(RBF_BEST_OUT_FILE))

RBF_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_RBF_SVM_,$(OUT_NO))
RBF_ALL_FILE=$(addsuffix .csv,$(RBF_ALL_OUT_FILE))
###################################################################


######### Decision Tree files .pbs files generate #############
DT_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_Decision_Tree_,$(OUT_NO))
DT_IMP=$(addsuffix .csv,$(DT_IMP_OUT_FILE))

DT_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_Decision_Tree_,$(OUT_NO))
DT_BEST=$(addsuffix .csv,$(DT_BEST_OUT_FILE))

DT_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_Decision_Tree_,$(OUT_NO))
DT_ALL_FILE=$(addsuffix .csv,$(DT_ALL_OUT_FILE))
###################################################################


######### Random Forest files .pbs files generate #############
RF_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_Random_Forest_,$(OUT_NO))
RF_IMP=$(addsuffix .csv,$(RF_IMP_OUT_FILE))

RF_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_Random_Forest_,$(OUT_NO))
RF_BEST=$(addsuffix .csv,$(RF_BEST_OUT_FILE))

RF_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_Random_Forest_,$(OUT_NO))
RF_ALL_FILE=$(addsuffix .csv,$(RF_ALL_OUT_FILE))
###################################################################


######### XGBOOST files .pbs files generate #############
XG_IMP_OUT_FILE=$(addprefix data/temp/all_imp_features_results_XGBoost_,$(OUT_NO))
XG_IMP=$(addsuffix .csv,$(XG_IMP_OUT_FILE))

XG_BEST_OUT_FILE=$(addprefix data/temp/best_hp_results_XGBoost_,$(OUT_NO))
XG_BEST=$(addsuffix .csv,$(XG_BEST_OUT_FILE))

XG_ALL_OUT_FILE=$(addprefix data/temp/all_hp_results_XGBoost_,$(OUT_NO))
XG_ALL_FILE=$(addsuffix .csv,$(XG_ALL_OUT_FILE))
###################################################################


$(L1_BEST)\
$(L1_IMP)\
$(L1_ALL_FILE)\
$(L2_BEST)\
$(L2_IMP)\
$(L2_ALL_FILE)\
$(Logit_BEST)\
$(Logit_IMP)\
$(Logit_ALL_FILE)\
$(RBF_BEST)\
$(RBF_IMP)\
$(RBF_ALL_FILE)\
$(DT_BEST)\
$(DT_IMP)\
$(DT_ALL_FILE)\
$(RF_BEST)\
$(RF_IMP)\
$(RF_ALL_FILE)\
$(XG_IMP)\
$(XG_ALL_FILE)\
$(XG_BEST)	:	output.in.intermediate;

.INTERMEDIATE:	output.in.intermediate
output.in.intermediate:	data/baxter.0.03.subsample.shared\
					data/metadata.tsv\
					$(CODE)/generateAUCs.R\
					$(CODE)/model_pipeline.R\
					$(CODE)/model_interpret.R\
					$(CODE)/main.R\
					$(CODE)/model_selection.R
	qsub L2_Logistic_Regression.pbs\
	qsub L1_Linear_SVM.pbs\
	qsub L2_Linear_SVM.pbs\
	qsub RBF_SVM.pbs\
	qsub Decision_Tree.pbs\
	qsub Random_Forest.pbs\
	qsub XGBoost.pbs



$(PROC)/combined_best_hp_results_L1_Linear_SVM.csv\
$(PROC)/combined_all_im_features_results_L1_Linear_SVM.csv\
$(PROC)/combined_all_hp_results_L1_Linear_SVM.csv\
$(PROC)/combined_best_hp_results_L2_Logistic_Regression.csv\
$(PROC)/combined_all_im_features_results_L2_Logistic_Regression.csv\
$(PROC)/combined_all_hp_results_L2_Logistic_Regression.csv\
$(PROC)/combined_best_hp_results_L2_Linear_SVM.csv\
$(PROC)/combined_all_im_features_results_L2_Linear_SVM.csv\
$(PROC)/combined_all_hp_results_L2_Linear_SVM.csv\
$(PROC)/combined_best_hp_results_RBF_SVM.csv\
$(PROC)/combined_all_im_features_results_RBF_SVM.csv\
$(PROC)/combined_all_hp_results_RBF_SVM.csv\
$(PROC)/combined_best_hp_results_Decision_Tree.csv\
$(PROC)/combined_all_im_features_results_Decision_Tree.csv\
$(PROC)/combined_all_hp_results_Decision_Tree.csv\
$(PROC)/combined_best_hp_results_Random_Forest.csv\
$(PROC)/combined_all_im_features_results_Random_Forest.csv\
$(PROC)/combined_all_hp_results_Random_Forest.csv\
$(PROC)/combined_best_hp_results_XGBoost.csv\
$(PROC)/combined_all_im_features_results_XGBoost.csv\
$(PROC)/combined_all_hp_results_XGBoost.csv	:	input.in.intermediate;

.INTERMEDIATE:	input.in.intermediate
input.in.intermediate:	code/cat_csv_files_test.sh\
						output.in.intermediate
		bash code/cat_csv_files_test.sh



################################################################################
#
# Part 3: Figure and table generation
#
#	Run scripts to generate figures and tables
#
################################################################################

# Figure 1 shows the generalization performance of all the models tested.
$(FIGS)/Figure_1.pdf :	$(CODE)/functions.R\
						$(CODE)/Figure1.R\
						$(PROC)/combined_best_hp_results_L2_Logistic_Regression.csv\
						$(PROC)/combined_best_hp_results_L1_Linear_SVM.csv\
						$(PROC)/combined_best_hp_results_L2_Linear_SVM.csv\
						$(PROC)/combined_best_hp_results_RBF_SVM.csv\
						$(PROC)/combined_best_hp_results_Decision_Tree.csv\
						$(PROC)/combined_best_hp_results_Random_Forest.csv\
						$(PROC)/combined_best_hp_results_XGBoost.csv
	Rscript $(CODE)/Figure1.R

# Figure 2 shows the hyper-parameter tuning of all the models tested.
$(FIGS)/Figure_2.pdf :	$(CODE)/functions.R\
						$(CODE)/Figure2.R\
						$(PROC)/combined_all_hp_results_L2_Logistic_Regression.csv\
						$(PROC)/combined_all_hp_results_L1_Linear_SVM.csv\
						$(PROC)/combined_all_hp_results_L2_Linear_SVM.csv\
						$(PROC)/combined_all_hp_results_RBF_SVM.csv\
						$(PROC)/combined_all_hp_results_Decision_Tree.csv\
						$(PROC)/combined_all_hp_results_Random_Forest.csv\
						$(PROC)/combined_all_hp_results_XGBoost.csv
	Rscript $(CODE)/Figure2.R

# Table 1 is a summary of the properties of all the models tested.
#$(TABLES)/Table1.pdf :	$(PROC)/model_parameters.txt\#
#						$(TABLES)/Table1.Rmd\#
#						$(TABLES)/header.tex#
#	R -e "rmarkdown::render('$(TABLES)/Table1.Rmd', clean=TRUE)"
#	rm $(TABLES)/Table1.tex









################################################################################
#
# Part 4: Pull it all together
#
# Render the manuscript
#
################################################################################


#$(FINAL)/manuscript.% : 			\ #include data files that are needed for paper don't leave this line with a : \
#						$(FINAL)/mbio.csl\#
#						$(FINAL)/references.bib\#
#						$(FINAL)/manuscript.Rmd#
#	R -e 'render("$(FINAL)/manuscript.Rmd", clean=FALSE)'
#	mv $(FINAL)/manuscript.knit.md submission/manuscript.md
#	rm $(FINAL)/manuscript.utf8.md


#write.paper : $(TABLES)/table_1.pdf $(TABLES)/table_2.pdf\ #customize to include
#				$(FIGS)/figure_1.pdf $(FIGS)/figure_2.pdf\	# appropriate tables and
#				$(FIGS)/figure_3.pdf $(FIGS)/figure_4.pdf\	# figures
#				$(FINAL)/manuscript.Rmd $(FINAL)/manuscript.md\#
#				$(FINAL)/manuscript.tex $(FINAL)/manuscript.pdf
