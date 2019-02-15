#!/bin/bash
SEARCH_DIR=data/temp
FINAL_DIR=data/process
	# Keep the first line of File1 and remove the first line of all the others and combine

for model in "L1_Linear_SVM" "L2_Logistic_Regression" "L2_Linear_SVM" "RBF_SVM" "Decision_Tree" "Random_Forest" "XGBoost"
	do
		head -1 $SEARCH_DIR/all_hp_results_"$model"_1.csv  > $SEARCH_DIR/test_combined_all_hp_results_"$model".csv; tail -n +2 -q $SEARCH_DIR/all_hp_results_"$model"_*.csv >> $SEARCH_DIR/test_combined_all_hp_results_"$model".csv

		head -1 $SEARCH_DIR/best_hp_results_"$model"_1.csv  > $SEARCH_DIR/test_combined_best_hp_results_"$model".csv; tail -n +2 -q $SEARCH_DIR/best_hp_results_"$model"_*.csv >> $SEARCH_DIR/test_combined_best_hp_results_"$model".csv

		head -1 $SEARCH_DIR/all_imp_features_results_"$model"_1.csv > $SEARCH_DIR/test_combined_all_imp_features_results_"$model".csv; tail -n +2 -q $SEARCH_DIR/all_imp_features_results_"$model"_*.csv >> $SEARCH_DIR/test_combined_all_imp_features_results_"$model".csv

        mv $SEARCH_DIR/test_combined_all_hp_results_"$model".csv $FINAL_DIR/test_combined_all_hp_results_"$model".csv
		mv $SEARCH_DIR/test_combined_best_hp_results_"$model".csv $FINAL_DIR/test_combined_best_hp_results_"$model".csv
		mv $SEARCH_DIR/test_combined_all_imp_features_results_"$model".csv $FINAL_DIR/test_combined_all_imp_features_results_"$model".csv
	done
