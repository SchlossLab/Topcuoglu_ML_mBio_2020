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
# Part 1: Get the references
#
# We will need several reference files to complete the analyses including the
# SILVA reference alignment and RDP reference taxonomy. Note that this code
# assumes that mothur is in your PATH. If not (e.g. it's in code/mothur/, you
# will need to replace `mothur` with `code/mothur/mothur` throughout the
# following code.
#
################################################################################

# We want the latest greatest reference alignment and the SILVA reference
# alignment is the best reference alignment on the market. This version is from
# 132 and described at http://blog.mothur.org/2018/01/10/SILVA-v132-reference-files/
# We will use the SEED v. 132, which contain 12,083 bacterial sequences. This
# also contains the reference taxonomy. We will limit the databases to only
# include bacterial sequences.

$(REFS)/silva.seed.align :
	wget -N https://mothur.org/w/images/7/71/Silva.seed_v132.tgz
	tar xvzf Silva.seed_v132.tgz silva.seed_v132.align silva.seed_v132.tax
	mothur "#get.lineage(fasta=silva.seed_v132.align, taxonomy=silva.seed_v132.tax, taxon=Bacteria);degap.seqs(fasta=silva.seed_v132.pick.align, processors=8)"
	mv silva.seed_v132.pick.align $(REFS)/silva.seed.align
	rm Silva.seed_v132.tgz silva.seed_v132.*

$(REFS)/silva.v4.align : $(REFS)/silva.seed.align
	mothur "#pcr.seqs(fasta=$(REFS)/silva.seed.align, start=11894, end=25319, keepdots=F, processors=8)"
	mv $(REFS)/silva.seed.pcr.align $(REFS)/silva.v4.align

# Next, we want the RDP reference taxonomy. The current version is v10 and we
# use a "special" pds version of the database files, which are described at
# http://blog.mothur.org/2017/03/15/RDP-v16-reference_files/

$(REFS)/trainset16_022016.% :
	wget -N https://www.mothur.org/w/images/c/c3/Trainset16_022016.pds.tgz
	tar xvzf Trainset16_022016.pds.tgz trainset16_022016.pds
	mv trainset16_022016.pds/* $(REFS)/
	rm -rf trainset16_022016.pds
	rm Trainset16_022016.pds.tgz

################################################################################
#
# Part 2: Get and run data through mothur
#
#	Process fastq data through the generation of files that will be used in the
# overall analysis.
#
################################################################################

# Change stability to the * part of your *.files file that lives in data/raw/
BASIC_STEM = data/mothur/stability.trim.contigs.good.unique.good.filter.unique.precluster


# here we go from the raw fastq files and the files file to generate a fasta,
# taxonomy, and count_table file that has had the chimeras removed as well as
# any non bacterial sequences.

# Edit code/get_good_seqs.batch to include the proper name of your *files file
$(BASIC_STEM).denovo.uchime.pick.pick.count_table $(BASIC_STEM).pick.pick.fasta $(BASIC_STEM).pick.pds.wang.pick.taxonomy : code/get_good_seqs.batch\
					data/references/silva.v4.align\
					data/references/trainset16_022016.pds.fasta\
					data/references/trainset16_022016.pds.tax
	mothur code/get_good_seqs.batch;\
	rm data/mothur/*.map



# here we go from the good sequences and generate a shared file and a
# cons.taxonomy file based on OTU data

# Edit code/get_shared_otus.batch to include the proper root name of your files file
# Edit code/get_shared_otus.batch to include the proper group names to remove

$(BASIC_STEM).pick.pick.pick.opti_mcc.unique_list.shared $(BASIC_STEM).pick.pick.pick.opti_mcc.unique_list.0.03.cons.taxonomy : code/get_shared_otus.batch\
					$(BASIC_STEM).denovo.uchime.pick.pick.count_table\
					$(BASIC_STEM).pick.pick.fasta\
					$(BASIC_STEM).pick.pds.wang.pick.taxonomy
	mothur code/get_shared_otus.batch
	rm $(BASIC_STEM).denovo.uchime.pick.pick.pick.count_table
	rm $(BASIC_STEM).pick.pick.pick.fasta
	rm $(BASIC_STEM).pick.pds.wang.pick.pick.taxonomy;


# now we want to get the sequencing error as seen in the mock community samples

# Edit code/get_error.batch to include the proper root name of your files file
# Edit code/get_error.batch to include the proper group names for your mocks

$(BASIC_STEM).pick.pick.pick.error.summary : code/get_error.batch\
					$(BASIC_STEM).denovo.uchime.pick.pick.count_table\
					$(BASIC_STEM).pick.pick.fasta\
					$(REFS)/HMP_MOCK.v4.fasta
	mothur code/get_error.batch


################################################################################
#
# Part 3: Model analysis in python
#
#	Run scripts to perform all the models on the dataset and generate AUC values
#
################################################################################


$(PROC)/L2_Logistic_Regression.tsv	$(PROC)/L1_SVM_Linear_Kernel.tsv	$(PROC)/SVM_RBF.tsv	$(PROC)/L2_SVM_Linear_Kernel.tsv	$(PROC)/Random_Forest.tsv	$(PROC)/Decision_Tree.tsv	$(PROC)/XGBoost.tsv	:	data/baxter.0.03.subsample.shared\
						data/metadata.tsv\
						code/learning/main.py\
						code/learning/preprocess_data.py\
						code/learning/model_selection.py
	python3 code/learning/main.py



################################################################################
#
# Part 4: Figure and table generation
#
#	Run scripts to generate figures and tables
#
################################################################################


$(FIGS)/Figure1.pdf :	$(PROC)/L2_Logistic_Regression.tsv\
						$(PROC)/L1_SVM_Linear_Kernel.tsv\
						$(PROC)/L2_SVM_Linear_Kernel.tsv\
						code/learning/compareAUC.R
	R -e "source('code/learning/compareAUC.R')"


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
