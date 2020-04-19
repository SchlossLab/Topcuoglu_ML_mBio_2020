# Author: Begum Topcuoglu
# Date: 2018-06-06
#
######################################################################
# This script plots the abundances of top 20 logistic and random forest featurs
######################################################################


######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("cowplot","reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
######################################################################
#----------------- Call the functions we will use -----------------#
######################################################################

source("code/learning/functions.R")


######################################################################
#-------- Get the full abundance data and metadata -----------------#
######################################################################
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, Dx_Bin, fit_result)
# Read in OTU table and remove label and numOtus columns
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
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



######################################################################
#----------------- Define the functions we will use -----------------#
######################################################################

# ------------------- Re-organize feature importance  ----------------->
# This function:
#     1. Takes in a combined (100 split) feature rankings for each model) and the model name
#     2. Returns the top 5 ranked (1-5 with 1 being the highest ranked) OTUs (ranks of the OTU for 100 splits)
get_feature_ranked_files <- function(file_name, model_name){
  importance_data <- read_tsv(file_name)
  ranks <- get_interp_info(importance_data, model_name) %>%
    as.data.frame()
  return(ranks)
}

logit_features <- get_feature_ranked_files("data/process/combined_L2_Logistic_Regression_feature_ranking.tsv", "L2_Logistic_Regression")

# Grab the names of the top 5 OTUs in the order of their median rank  
logit_otus <- logit_features %>% 
  group_by(key) %>% 
  summarise(imp = median(rank)) %>% 
  arrange(-imp) 

# Names of the y-axis labels
logit_taxa_info <- read.delim('data/baxter.taxonomy', header=T, sep='\t') %>% 
  select(-Size) %>% 
  mutate(key=OTU) %>% 
  select(-OTU)

logit_taxa_otus <- inner_join(logit_otus, logit_taxa_info, by="key") %>% 
  mutate_if(is.character, str_to_upper) %>%
  mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub("(.*)_.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub(".*;","",taxa)) %>% 
  mutate(taxa=gsub("(.*)_.*","\\1",taxa)) %>% 
  mutate(taxa=gsub('[0-9]+', '', taxa)) %>% 
  mutate(taxa=str_remove_all(taxa, "[(100)]")) %>% 
  select(key, taxa, imp) %>% 
  unite(key, taxa, key, sep=" (") %>% 
  mutate(key = paste(key,")", sep="")) %>% 
  mutate(key=paste0(gsub('TU0*', 'TU ', key))) 


logit_srn_top_otu_abundances <- data %>% 
  filter(dx != "normal") %>% 
  select(logit_otus$key)

logit_les_abunds <- logit_srn_top_otu_abundances/10000 + 1e-4

logit_normal_top_otu_abundances <- data %>% 
  filter(dx == "normal") %>% 
  select(logit_otus$key)

logit_norm_abunds <- logit_normal_top_otu_abundances/10000 + 1e-4

logit_les_otus <- logit_otus$key
logit_les_tax <- logit_taxa_otus$key




# Define the function to get the  most important top 20 OTUs
top_20_otus <-  function(name){
  
  # Order the dataframe from smallest new_auc to largest.
  # Because the smallest new_auc means that that OTU decreased AUC a lot when permuted
  
  data_first_20 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"),
                              header=T,
                              sep='\t') %>%
    arrange(imp) %>%
    head(20)
  
  return(data_first_20)
}

rf_otus <- top_20_otus("Random_Forest") %>% 
  arrange(-imp)


# Names of the y-axis labels
rf_taxa_info <- read.delim('data/baxter.taxonomy', header=T, sep='\t') %>% 
  select(-Size) %>% 
  mutate(names=OTU) %>% 
  select(-OTU)

rf_taxa_otus <- inner_join(rf_otus, rf_taxa_info, by="names") %>% 
  mutate_if(is.character, str_to_upper) %>%
  mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub("(.*)_.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
  mutate(taxa=gsub(".*;","",taxa)) %>% 
  mutate(taxa=gsub("(.*)_.*","\\1",taxa)) %>% 
  mutate(taxa=gsub('[0-9]+', '', taxa)) %>% 
  mutate(taxa=str_remove_all(taxa, "[(100)]")) %>% 
  select(names, taxa, imp) %>% 
  unite(names, taxa, names, sep=" (") %>% 
  mutate(names = paste(names,")", sep="")) %>% 
  mutate(names=paste0(gsub('TU0*', 'TU ', names))) 


rf_srn_top_otu_abundances <- data %>% 
  filter(dx != "normal") %>% 
  select(as.character(rf_otus$names))

rf_les_abunds <- rf_srn_top_otu_abundances/10000 + 1e-4

rf_normal_top_otu_abundances <- data %>% 
  filter(dx == "normal") %>% 
  select(as.character(rf_otus$names))

rf_norm_abunds <- rf_normal_top_otu_abundances/10000 + 1e-4

rf_les_otus <- as.character(rf_otus$names)
rf_les_tax <- rf_taxa_otus$names

#Stripchart of abundances
tiff('submission/Figure_S8.tiff', units="in", width=6.8, height=9, res=300)
layout(matrix(2:1, nrow=2), widths=c(0.5,1))

# L2-logistic-regression
par(mar=c(4,12,1.8,1.8))
plot(1, type="n", ylim=c(0.5,length(logit_les_otus)*2-1), xlim=c(1e-4,1), log="x", ylab="", xlab="Relative Abundance (%)", xaxt="n", yaxt="n")
index <- 1
for(i in logit_les_otus){
  stripchart(at=index-0.35, jitter(logit_norm_abunds[,i], amount=1e-5), pch=21, bg="lightskyblue", method="jitter", jitter=0.2, add=T, cex=0.8, lwd=0.5)
  stripchart(at=index+0.35, jitter(logit_les_abunds[,i], amount=1e-5), pch=21, bg="red1", method="jitter", jitter=0.2, add=T, cex=0.8, lwd=0.5)
  segments(median(logit_norm_abunds[,i]),index-0.75,median(logit_norm_abunds[,i]),index+0.05, lwd=5)
  segments(median(logit_les_abunds[,i]),index+0.75,median(logit_les_abunds[,i]),index-0.05, lwd=5)
  index <- index + 2
}
legend("topright", legend=c("SRN", "Normal"), pch=c(21, 21), pt.bg=c("red3","lightskyblue"))
axis(1, at=c(1e-4, 1e-3, 1e-2, 1e-1, 1), label=c("0", "0.1", "1", "10", "100"))
mtext(logit_les_tax, at=1:length(logit_les_otus)*2-1, side=2, las=1, adj=1, line=0.5)
mtext('B - L2 logistic regression',cex=1.5, font=8, side=3, adj= 0, line=0.5)

# random forest
par(mar=c(4,12,1.8,1.8))
plot(1, type="n", ylim=c(0.5,length(rf_les_otus)*2-1), xlim=c(1e-4,1), log="x", ylab="", xlab="", xaxt="n", yaxt="n")
index <- 1
for(i in rf_les_otus){
  stripchart(at=index-0.35, jitter(rf_norm_abunds[,i], amount=1e-5), pch=21, bg="lightskyblue", method="jitter", jitter=0.2, add=T, cex=0.8, lwd=0.5)
  stripchart(at=index+0.35, jitter(rf_les_abunds[,i], amount=1e-5), pch=21, bg="red1", method="jitter", jitter=0.2, add=T, cex=0.8, lwd=0.5)
  segments(median(rf_norm_abunds[,i]),index-0.75,median(rf_norm_abunds[,i]),index+0.05, lwd=5)
  segments(median(rf_les_abunds[,i]),index+0.75,median(rf_les_abunds[,i]),index-0.05, lwd=5)
  index <- index + 2
}
legend("topright", legend=c("SRN", "Normal"), pch=c(21, 21), pt.bg=c("red1","lightskyblue"))
axis(1, at=c(1e-4, 1e-3, 1e-2, 1e-1, 1), label=c("0", "0.1", "1", "10", "100"))
mtext(rf_les_tax, at=1:length(rf_les_otus)*2-1, side=2, las=1, adj=1, line=0.5)
mtext('A - Random forest',cex=1.5, font=8, side=3, adj= 0, line=0.2)
dev.off()

