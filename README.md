### Overview

	project
	|- README          # the top level description of content (this doc)
	|- CONTRIBUTING    # instructions for how to contribute to your project
	|- LICENSE         # the license for this project
	|
	|- data           # raw and primary data, are not changed once created
	|
	|- code/          # any programmatic code
	|
	|- results        # all output from workflows and analyses
	| |- tables/      # text version of tables to be rendered with kable in R
	| |- figures/     # graphs, likely designated for manuscript figures
	| +- pictures/    # diagrams, images, and other non-graph graphics


### How to regenerate this repository

#### Dependencies and locations
* Python 3.6.5
* Latest PyTorch (pip3 install torch torchvision)

#### Clone the repository

```
git clone https://github.com/BTopcuoglu/DeepLearning
```
