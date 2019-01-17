#!/bin/bash          
SEARCH_DIR=data/process/parameter_search

# concatenate output from all parameter searches into one file 
# with only the header from the firsrt file
# and only lines with 1 at the end of the run
{ head -n 1 $SEARCH_DIR/parameter_search_HL_070617_1.modelRunHistory.csv & 
	awk '/1"$/' $SEARCH_DIR/*.modelRunHistory.csv ; } > data/process/cat_parameter_sets.csv