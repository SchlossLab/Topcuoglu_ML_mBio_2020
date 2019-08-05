RANK=feature_ranking
SCORE=feature_scores

# Define the directories we will use in the script
SEARCH_DIR=data/temp
FINAL_DIR=data/process

# 1. Keep the first line of File0 and remove the first line of all the other files (File[0-99]) and
#		output it to the FINAL_DIR location
cp $SEARCH_DIR/L1_Linear_SVM_"$RANK"_1.tsv $FINAL_DIR/combined_L1_Linear_SVM_"$RANK".tsv
cp $SEARCH_DIR/L2_Linear_SVM_"$RANK"_101.tsv $FINAL_DIR/combined_L2_Linear_SVM_"$RANK".tsv
cp $SEARCH_DIR/L2_Logistic_Regression_"$RANK"_201.tsv $FINAL_DIR/combined_L2_Logistic_Regression_"$RANK".tsv


#	2. Append the other files to the end, but we want to be sure to ignore the 0 file since we don't
#		want it printed twice
#        "tail -n +2" makes tail print lines from 2nd line to the end
#        "-q" tells it to not print the header with the file name
#        ">>" adds all the tail stuff from every file to the combined file
tail -n +2 -q $SEARCH_DIR/L1_Linear_SVM_"$RANK"_{2..100}.tsv >> $FINAL_DIR/combined_L1_Linear_SVM_"$RANK".tsv
tail -n +2 -q $SEARCH_DIR/L2_Linear_SVM_"$RANK"_{102..200}.tsv >> $FINAL_DIR/combined_L2_Linear_SVM_"$RANK".tsv
tail -n +2 -q $SEARCH_DIR/L2_Logistic_Regression_"$RANK"_{202..300}.tsv >> $FINAL_DIR/combined_L2_Logistic_Regression_"$RANK".tsv

# 1. Keep the first line of File0 and remove the first line of all the other files (File[0-99]) and
#		output it to the FINAL_DIR location

#cp $SEARCH_DIR/L1_Linear_SVM_"$SCORE"_1.tsv $FINAL_DIR/combined_L1_Linear_SVM_"$SCORE".tsv
#cp $SEARCH_DIR/L2_Linear_SVM_"$SCORE"_101.tsv $FINAL_DIR/combined_L2_Linear_SVM_"$SCORE".tsv
#cp $SEARCH_DIR/L2_Logistic_Regression_"$SCORE"_200.tsv $FINAL_DIR/combined_L2_Logistic_Regression_"$SCORE".tsv


#	2. Append the other files to the end, but we want to be sure to ignore the 0 file since we don't
#		want it printed twice
#        "tail -n +2" makes tail print lines from 2nd line to the end
#        "-q" tells it to not print the header with the file name
#        ">>" adds all the tail stuff from every file to the combined file
#tail -n +2 -q $SEARCH_DIR/L1_Linear_SVM_"$SCORE"_{2..100}.tsv >> $FINAL_DIR/combined_L1_Linear_SVM_"$SCORE".tsv
#tail -n +2 -q $SEARCH_DIR/L2_Linear_SVM_"$SCORE"_{101..199}.tsv >> $FINAL_DIR/combined_L2_Linear_SVM_"$SCORE".tsv
#tail -n +2 -q $SEARCH_DIR/L2_Logistic_Regression_"$SCORE"_{200..299}.tsv >> $FINAL_DIR/combined_L2_Logistic_Regression_"$SCORE".tsv