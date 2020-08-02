% Copyright (C) 2020 Jacopo Cirrone


This repository contains PhenoPredict, 

a python developed Method for predicting phenotypes from gene expression data as well as to predict causal genes.

The aim of this machine learning strategy is to create causality models for the phenotypes. In other words, the causality models are constructed to predict phenotypes from gene expression data and then learning the effect of genes and TFs on phenotypes. This is valuable because if, say, any gene g is differentially expressed when plants have a positive phenotypic trait such as high yield, we want to infer g so that plants can be transformed to reach this desired trait.


In a case study,
we used Rice lab data from Nipponbare up to three weeks to produce non-linear (random forest) models that predict biomass at the end of those three weeks. 

We then applied the same model to predict both biomass and yield across different genotypes in the field (2x2 for 19 genotypes) at two months. 

Surprisingly, the model predicted biomass and yield very well. Specifically, the biomass model for Nipponbare at three weeks can be used on two-month old IR83383, PSBRC, IR74371_54, IR87707, PR106, Nipponbare, IR74371_70, IR64 (genotypes) to predict their biomasses and on two-month old Palawan, IR20, IR83380, IR74371_70, PSBRC, IR83383, IR74371_54 (genotypes) to predict their yields.

In addition, the list of transcription factors that are shown to be important to predict biomass in Nipponbare include genes that are present in most of these other genotypes. Some of these already have a known role in positive phenotypes. Our model both 
(i) suggests which transcription factors to test for at three weeks to predict high biomass and high yield at two months and 
(ii) which transcription factors to over-express and which ones to repress in transformed plants to achieve higher biomass and yield. 



## Setup and Run PhenoPredict

To run PhenoPredict, Miniconda or Anaconda must be previously installed (Anaconda: https://www.anaconda.com/distribution/#download-section)

Clone the codebase:

```
git clone https://github.com/jacirrone/PhenoPredict.git
```

Enter the PhP directory:

```
cd PhP/
```
```

To install PhenoPredict, install the following libraries as follows:

```
pip install sklearn pandas xgboost
```

As example to run PhenoPredict, invoke the corresponding script:
 
python RicePredictor3xg_biomass.py




## Required data for PhenoPredict

The Datasets directory, "PhP/data/", contains the input files

training_data.csv
-----------------
expression values for the training data; first row is the phenotype to model from the training data;

Obtain expression data and save it as a csv file "training_data.csv" of [Genes x Samples]

test_data.csv
-----------------
expression values for the test data;

Obtain expression data and save it as a csv file "test_data.csv" of [Genes x Samples]

test_phenotype.csv
-----------------
phenotype values for the test data;

Obtain phenotype data and save it as a csv file "test_phenotype.csv" of [Phenotypes x Samples]



## Run PhenoPredict

Enter the PhP directory

```
cd PhP/
```


To use PhenoPredict with Random Forest
```
python RicePredictor3xg_biomass.py -em RF

```


To use PhenoPredict with XGBoosting
```
python RicePredictor3xg_biomass.py -em XG

```

