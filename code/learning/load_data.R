######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("reshape2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

######################################################################
#----------------- Load OTU table and MetaData -----------------#
######################################################################
# Read in metadata and select only sample Id and diagnosis columns
meta <- read.delim('../data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, dx, Dx_Bin, fit_result)


# Read in OTU table and remove label and numOtus columns
shared <- read.delim('../data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
  select(-label, -numOtus)

# Merge metadata and OTU table.
# Group advanced adenomas and cancers together as cancer and normal, high risk normal and non-advanced adenomas as normal
# Then remove the sample ID column
data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
  mutate(dx = case_when(
    Dx_Bin== "Adenoma" ~ "normal",
    Dx_Bin== "Normal" ~ "normal",
    Dx_Bin== "High Risk Normal" ~ "normal",
    Dx_Bin== "adv Adenoma" ~ "cancer",
    Dx_Bin== "Cancer" ~ "cancer"
  )) %>%
  select(-sample, -Dx_Bin, -fit_result) %>%
  drop_na()
# We want the diagnosis column to be a factor
data$dx <- factor(data$dx)
