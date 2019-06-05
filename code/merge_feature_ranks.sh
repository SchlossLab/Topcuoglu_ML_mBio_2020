# Define the model and data type to use
MODEL=$1	
#   1.  L1_Linear_SVM
#   2.  L2_Linear_SVM
#   3.  L2_Logistic_Regression

DATA=feature_ranking

# Define the directories we will use in the script
SEARCH_DIR=data/process
FINAL_DIR=data/process

# 1. Keep the first line of File0 and remove the first line of all the other files (File[0-99]) and
#		output it to the FINAL_DIR location
cp $SEARCH_DIR/"$MODEL"_"$DATA"_1.csv $FINAL_DIR/combined_"$MODEL"_"$DATA".csv

#	2. Append the other files to the end, but we want to be sure to ignore the 0 file since we don't
#		want it printed twice
#        "tail -n +2" makes tail print lines from 2nd line to the end
#        "-q" tells it to not print the header with the file name
#        ">>" adds all the tail stuff from every file to the combined file
tail -n +2 -q $SEARCH_DIR/"$MODEL"_"$DATA"_{1..300}.csv >> $FINAL_DIR/combined_"$MODEL"_"$DATA".csv