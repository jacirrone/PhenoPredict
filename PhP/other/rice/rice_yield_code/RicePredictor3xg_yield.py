import os

import sys

import pandas as pd

import numpy as np

import argparse as argp

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.datasets import make_regression

# from numpy.random import *

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

import scipy

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['PYTHONHASHSEED']='42'
#np.random.seed(42)


def main(*args):

	#np.random.seed(42)
	ensemble_method = args[0]

	if ensemble_method != "RF" and ensemble_method != "XG" and ensemble_method != "AB":
		print("*****************************************")
		print("Error: To run PhenoPredict properly choose one of the ensemble method between RF or XG")
		exit(1)

	if ensemble_method == "XG":
		from xgboost import XGBRegressor

	
	ntrees = 6000#3000
	niter = 5000
	filename_output = str(ensemble_method+str(ntrees)+"Trees.txt")
	orig_stdout = sys.stdout
	f = open(filename_output, 'w')
	sys.stdout = f
	print("Number of trees:", ntrees)#, "and Gradient Boosting Regressor"
	#Analysis
	#797 genes from the field data are all within the 6979 d.e. of the lab data

	#:::::::FIELD DATA : 3 replicated for 4 different conditions (LowWaterHighNitrogen, HWHN, LWLN, HWLN) 
	#for 19 different genotypes = 228 single data points

	#Read phenotypes field data
	phenotype_data = pd.read_csv("./data/Field_phenotype.csv")
	#Set of features: each class is a different set of features; all the genes as features
	phenotype_data.drop("model", axis=1, inplace=True)
	phenotype_names = phenotype_data["Gene"].values
	phenotype_data.set_index("Gene", inplace=True)
	phenotype_data = phenotype_data.transpose()
	print("phenotype_names.shape", phenotype_names.shape)#(228,8)

	#FIELD GENE EXPRESSION DATA
	#for 19 different genotypes = 228 single data points
	geneexpression_data = pd.read_csv("./data/Field_data.csv")

	#geneexpression_data (6979, 228)
	geneexpression_data.set_index("Gene", inplace=True)
	geneexpression_data = geneexpression_data.transpose()
	print("Field data - all genes expression dimension: ", geneexpression_data.values.shape)


	# water_genes = geneexpression_data[(geneexpression_data.model=="W")].copy()#196
	# mole_genes = geneexpression_data[(geneexpression_data.model=="N")].copy()#123
	# molarity_genes = geneexpression_data[(geneexpression_data.model=="N/W")].copy()#137
	# sinergy_genes = geneexpression_data[(geneexpression_data.model=="NxW")].copy()#341

	# water_genes.drop("model", axis=1, inplace=True)
	# water_genes.set_index("Gene", inplace=True)
	# water_genes = water_genes.transpose()
	# print("water genes expression dimension: ", water_genes.values.shape


	# mole_genes.drop("model", axis=1, inplace=True)
	# mole_genes.set_index("Gene", inplace=True)
	# mole_genes = mole_genes.transpose()
	# print("mole genes expression dimension: ", mole_genes.values.shape


	# molarity_genes.drop("model", axis=1, inplace=True)
	# molarity_genes.set_index("Gene", inplace=True)
	# molarity_genes = molarity_genes.transpose()
	# print("molarity genes expression dimension: ", molarity_genes.values.shape


	# sinergy_genes.drop("model", axis=1, inplace=True)
	# sinergy_genes.set_index("Gene", inplace=True)
	# sinergy_genes = sinergy_genes.transpose()
	# print("sinergy genes expression dimension: ", sinergy_genes.values.shape



	#::::::LAB DATA
	#Different levels of nitrogen and water. Total of 16 conditions times 3 reps 48 samples 
	#but total is 45 (some reps are missing)
	lab_data = pd.read_csv("./data/Lab_data.csv")
	lab_data.set_index("Gene", inplace=True)
	#lab_data.index.name = "Gene"
	biomass = lab_data.loc["Biomass"]
	lab_data.drop("Biomass", axis=0, inplace=True)
	conds_lab_data = lab_data.columns

	genes_features_filtered = pd.Series(list(set(geneexpression_data.columns).intersection(set(lab_data.index.values)))).values
	#genes_features_filtered = water_genes.columns.values#mole_genes #molarity_genes #sinergy_genes #
	lab_data_filtered = lab_data.loc[genes_features_filtered].copy()


	lab_data_filtered = lab_data_filtered.transpose()
	print("lab_data_filtered.shape", lab_data_filtered.shape)#(45, 6979)

	#X [n_samples, n_features]
	#y [n_samples, num_genes]
	#Features
	genes_features_filtered = genes_features_filtered#genes_features_filtered[indxes_best_features]#genes_features_filtered[indxes_best_features]#genes_features_filtered



	X = lab_data_filtered.loc[:, genes_features_filtered]#lab_data_filtered_4conds.values#lab_data_filtered.values#
	y = biomass.values#biomass_4conds.values#[i * 100 for i in biomass.values]

	#y = np.ravel(y)


	# indx_phenotype_notna = np.isfinite(phenotype_data.loc[:, "Biomass"])
	# y_test_biomass = phenotype_data.loc[indx_phenotype_notna, "Biomass"]


	# #Run randomized search using the most important features from the model using all the data
	# treeEstimator = RandomForestRegressor(n_estimators=ntrees, max_features="sqrt", n_jobs=-1, oob_score = True)

	# n_iter_search = 30

	# # specify parameters and distributions to sample from
	# param_dist = {"n_estimators": [ntrees],
	# "min_samples_split": sp_randint(2, 30),
	# "max_depth" : [1, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
	# "min_samples_leaf" : [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 65, 80],
	# "oob_score": [True],
	# "max_features": ["sqrt"],
	# "n_jobs": [-1]}
	# #"max_depth"         : [2, 5, 10, 20, 25],
	# #"min_samples_split" : [2, 4, 6, 8, 10, 15],
	# #"max_depth" : [1, 5, 10, 15, 20, 25, 30],
	# #"min_samples_leaf" : [1, 2, 4, 6, 8, 10]}


	# random_search = RandomizedSearchCV(treeEstimator, param_distributions=param_dist,
	#                    n_iter=n_iter_search, n_jobs=1)#cv=2
	# random_search.fit(X, y)#X_all_data, y_all_data)#(X_std, y_std)

	# print(random_search.best_params_

	#45x797 lab data

	#LAB DATA CONDITIONS
	#N0.625; N1.25; N2.5; N5.0
	#W12.5; W25; W50; W100
	#Lab data: W12.5_N1.25_1 (for example) means that the water level (W) was 12.5%, 
	#the nitrogen (N) amount was 1.25mM and it is one of 3 replicates


	# FROM FILED DATA CONDITIONS TO LAB DATA CONDITIONS
	# W - {12.5, 100}
	# N - {0.625, 5.0}
	# LWHN (12.5, 5.0)
	# HWHN (100, 5.0)
	# LWLN (12.5, 0.625)
	# HWLN (100, 0.625)

	# LWLN - 'W12.5_N0.625_1', u'W12.5_N0.625_2', u'W12.5_N0.625_3'
	# LWHN - 'W12.5_N5.0_1', u'W12.5_N5.0_2', u'W12.5_N5.0_3'
	# HWLN - 'W100_N0.625_1', u'W100_N0.625_2', u'W100_N0.625_3'
	# HWHN - 'W100_N5.0_1',u'W100_N5.0_2', u'W100_N5.0_3'

	#FIELD DATA Nipponbare
	# 4s-14 Nipponbare  LWHN
	# 4s-23 Nipponbare  LWHN
	# 4s-54 Nipponbare  LWHN
	# 4w-14 Nipponbare  HWHN
	# 4w-23 Nipponbare  HWHN
	# 4w-54 Nipponbare  HWHN
	# 9s-14 Nipponbare  LWLN
	# 9s-23 Nipponbare  LWLN
	# 9s-54 Nipponbare  LWLN
	# 9w-14 Nipponbare  HWLN
	# 9w-23 Nipponbare  HWLN
	# 9w-54 Nipponbare  HWLN

	#Biomass field in grams
	#Biomass lab in grams too


	# # LWLN
	# # LWHN
	# # HWLN
	# # HWHN 
	# lab_indx_for_field_conds = ['W12.5_N0.625_1', 'W12.5_N0.625_2', 'W12.5_N0.625_3', 'W12.5_N5.0_1', 'W12.5_N5.0_2', 'W12.5_N5.0_3', 'W100_N0.625_1', 'W100_N0.625_2', 'W100_N0.625_3', 'W100_N5.0_1', 'W100_N5.0_2', 'W100_N5.0_3']
	# field_indxes_nipp = ["9s-14", "9s-23", "9s-54", "4s-14", "4s-23", "4s-54", "9w-14", "9w-23", "9w-54", "4w-14", "4w-23", "4w-54"]


	#nipp
	field_indxes_nipp = ["9s-14", "9s-23", "9s-54", "4s-14", "4s-23", "4s-54", "9w-14", "9w-23", "9w-54", "4w-14", "4w-23", "4w-54"]
	#bg348
	field_indxes_bg348 = ["9s-15", "9s-30", "9s-47", "4s-15", "4s-30", "4s-47", "9w-15", "9w-30", "9w-47", "4w-15", "4w-30", "4w-47"]
	#bg380
	field_indxes_bg380 = ["9s-06", "9s-29", "9s-55", "4s-06", "4s-29", "4s-55", "9w-06", "9w-29", "9w-55", "4w-06", "4w-29", "4w-55"]
	#Beonjo
	field_indxes_beonjo = ["9s-16", "9s-36", "9s-58", "4s-16", "4s-36", "4s-58", "9w-16", "9w-36", "9w-58", "4w-16", "4w-36", "4w-58"]

	field_indxes_bg90 = ["9s-03", "9s-35", "9s-45", "4s-03", "4s-35", "4s-45", "9w-03", "9w-35", "9w-45", "4w-03", "4w-35", "4w-45"]
	field_indxes_hagino_mochi = ["9s-05", "9s-24", "9s-60", "4s-05", "4s-24", "4s-60", "9w-05", "9w-24", "9w-60","4w-05", "4w-24", "4w-60"]
	field_indxes_ir20 = ["9s-18", "9s-32", "9s-53", "4s-18", "4s-32", "4s-53", "9w-18", "9w-32", "9w-53", "4w-18", "4w-32", "4w-53"]
	field_indxes_ir64 = [ "9s-19", "9s-31", "9s-57", "4s-19", "4s-31", "4s-57", "9w-19", "9w-31", "9w-57", "4w-19", "4w-31", "4w-57"]
	field_indxes_ir74371_54 = [ "9s-17", "9s-27", "9s-43", "4s-17", "4s-27", "4s-43", "9w-17", "9w-27", "9w-43", "4w-17", "4w-27", "4w-43"]
	field_indxes_ir74371_70 = [ "9s-10", "9s-26", "9s-56", "4s-10", "4s-26", "4s-56", "9w-10", "9w-26", "9w-56", "4w-10", "4w-26", "4w-56"]
	field_indxes_ir83380 = [ "9s-04", "9s-34", "9s-50", "4s-04", "4s-34", "4s-50", "9w-04", "9w-34", "9w-50", "4w-04", "4w-34", "4w-50"]
	field_indxes_ir83383 = [ "9s-09", "9s-25", "9s-51", "4s-09", "4s-25", "4s-51", "9w-09", "9w-25", "9w-51", "4w-09", "4w-25", "4w-51"]
	field_indxes_ir83388 = [ "9s-08", "9s-21", "9s-52", "4s-08", "4s-21", "4s-52", "9w-08", "9w-21", "9w-52", "4w-08", "4w-21", "4w-52"]
	field_indxes_ir87707 = [ "9s-01", "9s-38", "9s-49", "4s-01", "4s-38", "4s-49", "9w-01", "9w-38", "9w-49", "4w-01", "4w-38", "4w-49"]
	field_indxes_palawan = ["9s-02", "9s-28", "9s-46", "4s-02", "4s-28", "4s-46", "9w-02", "9w-28", "9w-46", "4w-02", "4w-28", "4w-46"]
	field_indxes_pr106 = ["9s-11", "9s-22", "9s-59", "4s-11", "4s-22", "4s-59", "9w-11", "9w-22", "9w-59", "4w-11", "4w-22", "4w-59"]
	field_indxes_psbrc = ["9s-13", "9s-39", "9s-44", "4s-13", "4s-39", "4s-44", "9w-13", "9w-39", "9w-44", "4w-13", "4w-39", "4w-44"]
	field_indxes_tainung = ["9s-12", "9s-40", "9s-48", "4s-12", "4s-40", "4s-48", "9w-12", "9w-40", "9w-48", "4w-12", "4w-40", "4w-48"]
	field_indxes_yabani = ["9s-07", "9s-33", "9s-41", "4s-07", "4s-33", "4s-41", "9w-07", "9w-33", "9w-41", "4w-07", "4w-33", "4w-41"]

	#ONLY 2 NAN Biomass VALUES: 9s-59 and 4s-49

	genotypes = [field_indxes_nipp, field_indxes_bg348, field_indxes_bg380, field_indxes_beonjo, field_indxes_bg90, field_indxes_hagino_mochi, 
	field_indxes_ir20, field_indxes_ir64, field_indxes_ir74371_54, field_indxes_ir74371_70, field_indxes_ir83380, field_indxes_ir83383, 
	field_indxes_ir83388, field_indxes_ir87707, field_indxes_palawan, field_indxes_pr106, field_indxes_psbrc, field_indxes_tainung, field_indxes_yabani]

	number_seeds = 5

	df_corr2 = pd.DataFrame(np.c_[np.ones(number_seeds), 
	                   np.ones(number_seeds), 
	                   np.ones(number_seeds), 
	                   np.ones(number_seeds),np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds),
	                   np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds),
	                   np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds),
	                   np.ones(number_seeds), np.ones(number_seeds), np.ones(number_seeds)], 
	                  columns=["Nipponbare","BG348","BG380", "Beonjo", "BG90", "Hagino Mochi", 
	                  "IR20", "IR64","IR74371_54", "IR74371_70","IR83380", "IR83383",
	                  "IR83388", "IR87707","Palawan", "PR106","PSBRC", "Tainung","Yabani"])


	print("X.shape matrix [n_samples, n_features-n_genes] dimension", X.shape)
	print("y.shape [n_samples] dimension", y.shape)
	#print("X_test.shape", X_test.shape
	
	for l in range(0, number_seeds):

		if ensemble_method == "XG":

			# num_boost_round is the same as n_estimators

			clf = XGBRegressor(
			#n_estimators = ntrees,
			eval_metric = 'rmse',
			#nthread = 30,
			eta = 0.1,
			num_boost_round = 80,
			max_depth = 5,
			subsample = 0.8,
			colsample_bytree = 0.5,
			silent = 1,
			#seed = 42,
			early_stopping_rounds=200
			)


			#params to regularize more given the small dataset
			parameters = {
			'num_boost_round': [10, 25, 50, 100, 200, 300, 500, 1000, 3000, 6000],
			'eta': [0.01, 0.04, 0.07, 0.1, 0.2, 0.3], #learning rate
			'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
			'subsample': [0.6, 0.7, 0.8, 0.9],
			'colsample_bytree': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]#,
			#'min_child_weight': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 50] 
			}


			# parameters = {
			#     'num_boost_round': [10, 25, 50, 100, 200, 300, 500, 1000],
			#     'eta': [0.005, 0.01, 0.04, 0.07, 0.1, 0.2, 0.3, 0.5], #learning rate
			#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 50, 100],
			#     'subsample': [0.6, 0.7, 0.8, 0.9],
			#     'colsample_bytree': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
			#     'min_child_weight': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 50] 
			# }

			# lambda reg param in similarity score
			# gamma reg param for pruning

			#Initial params range found.
			# parameters = {
			#     'num_boost_round': [10, 25, 50],
			#     'eta': [0.05, 0.1, 0.3], #learning rate
			#     'max_depth': [3, 4, 5],
			#     'subsample': [0.9, 1.0],
			#     'colsample_bytree': [0.9, 1.0],
			# }
			#np.random.seed(42)

			random_search = RandomizedSearchCV(clf, parameters, n_jobs=-1, cv=5, n_iter=niter)#, random_state=42)
			#random_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=10)


			start2 = time.time()
			random_search.fit(X, y)


			best_parameters = random_search.best_params_
			# for param_name in sorted(best_parameters.keys()):
			#     print("%s: %r" % (param_name, best_parameters[param_name]))
			end2 = time.time()
			print('time elapsed: ' + str(end2-start2))           


			treeEstimator2 = XGBRegressor(**best_parameters)
			treeEstimator2.fit(X, y)


			#XGBRegressor has no attibute oob_score_
			#oobscore2 = 0
			#print("GB single ", ntrees, "num of trees oob score", treeEstimator2.oob_score_

		elif ensemble_method == "RF":

			#np.random.seed(42)


			params = {'oob_score': True, 
			'min_samples_leaf': 2, 'n_jobs': -1, 'n_estimators': ntrees, 
			#'min_samples_split': 10, 
			'max_features': 'sqrt', 
			#'max_depth': 6
			}

			treeEstimator2 = RandomForestRegressor(**params)
			treeEstimator2.fit(X, y)
			#oobscore2 = treeEstimator.oob_score_

		elif ensemble_method == "AB":

			print("Hyper parameters optimization for AdaBoost")

			#np.random.seed(42)

			param_dist = {
			'n_estimators': [10, 25, 50, 100, 200],#, 300, 500, 1000, 3000, 6000],
			'learning_rate' : [0.01,0.05,0.1,0.3,1],
			#'loss' : ['linear', 'square', 'exponential']
			 }

			if niter > 120:
				niter = 120

			random_search = GridSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=1)),
			 param_grid = param_dist,
			 cv=10,
			 n_jobs=-1#,
			 #n_iter=niter
			 )

			start2 = time.time()
			random_search.fit(X, y)


			best_parameters = random_search.best_params_
			print("Best score", random_search.best_score_)
			# for param_name in sorted(best_parameters.keys()):
			#     print("%s: %r" % (param_name, best_parameters[param_name]))
			end2 = time.time()
			print('time elapsed: ' + str(end2-start2))           


			treeEstimator2 = AdaBoostRegressor(**best_parameters)
			print("training score", treeEstimator2.fit(X, y).score(X,y))
			




		for g_id, genotype in enumerate(genotypes):
			field_genexp_data_single_genotype = geneexpression_data.loc[genotype, genes_features_filtered]#indx_phenotype_notna, genes_features_filtered]#field_indxes_nipp, genes_features_filtered]
			y_test_biomass_field = phenotype_data.loc[genotype, "Grain Yield"]
			X_test = field_genexp_data_single_genotype


			y_pred = treeEstimator2.predict(X_test)

			#print('Iteration')
			#print(l, g_id, genotype, y_pred)

			#corr_tmp, pval_corr = scipy.stats.pearsonr(y_test_biomass_field,y_pred)

			LWLN_pred = np.mean(y_pred[[0,1,2]])
			LWHN_pred = np.mean(y_pred[[3,4,5]])
			HWLN_pred = np.mean(y_pred[[6,7,8]])
			HWHN_pred = np.mean(y_pred[[9,10,11]])
			y_pred_avg = [LWHN_pred, LWLN_pred, HWLN_pred, HWHN_pred]
			y_pred_avg = [i for i in y_pred_avg]

			LWLN_field = np.mean(y_test_biomass_field[[0,1,2]])
			LWHN_field = np.mean(y_test_biomass_field[[3,4,5]])
			HWLN_field = np.mean(y_test_biomass_field[[6,7,8]])
			HWHN_field = np.mean(y_test_biomass_field[[9,10,11]])
			y_test_biomass_field_avg = [LWHN_field, LWLN_field, HWLN_field, HWHN_field]
			y_test_biomass_field_avg = [i for i in y_test_biomass_field_avg]

			# print("y_test_biomass_field_avg")
			# print(y_test_biomass_field_avg)
			corr_tmp_avg, pval_avg = scipy.stats.pearsonr(y_test_biomass_field_avg, y_pred_avg)
			# print(corr_tmp_avg, "corr_tmp_avg")
			df_corr2.iloc[l, g_id] = corr_tmp_avg



	width = 0.85

	colors=["red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue"]



	#print(Results for Thousands of Trees Forest
	df_corr2 = df_corr2.reindex(df_corr2.mean().sort_values(ascending=False).index, axis=1)
	value = df_corr2.mean()
	std = df_corr2.std()
	#print('STD ERROR')
	#print(std)
	width = 0.85
	colors=["red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue"]
	plt.bar(range(len(df_corr2.columns)), value, width, yerr=std, color=colors, error_kw=dict(lw=0.4, capsize=2, capthick=0.4), alpha=0.5, align='center')
	plt.xlim([-1, len(df_corr2.columns)])
	plt.xticks(range(len(df_corr2.columns)), df_corr2.columns, rotation='vertical', fontsize=9)#, step=0.5)
	plt.yticks(np.arange(-1, 1.1, 0.1))
	plt.ylabel('Correlation ')
	plt.title('Field Yield Prediction across Genotypes using Lab Nipponbare Model')
	plt.savefig("YieldPredictionResults_"+ensemble_method+str(ntrees)+"Trees.pdf", bbox_inches='tight')
	plt.close()
	df_corr_filter2 = df_corr2.loc[:, (df_corr2.mean() > 0.5)] #& (df_corr2.std() < 0.3)]
	value = df_corr_filter2.mean()
	std = df_corr_filter2.std()
	width = 0.85
	colors=["red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue", "purple", "red", "green", "blue"]
	colors = colors[:len(df_corr_filter2.columns)]
	plt.bar(range(len(df_corr_filter2.columns)), value, width, yerr=std, color=colors,error_kw=dict(lw=0.4, capsize=2, capthick=0.4), alpha=0.5, align='center')
	plt.xlim([-1, len(df_corr_filter2.columns)])
	plt.xticks(range(len(df_corr_filter2.columns)), df_corr_filter2.columns, rotation='vertical', fontsize=9)#, step=0.5)
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.ylabel('Correlation ')
	plt.title('Field Yield Prediction across Genotypes using Lab Nipponbare Model')
	plt.savefig("YieldPredictionResults_"+ensemble_method+str(ntrees)+"Trees_TOP.pdf", bbox_inches='tight')
	plt.close()
	df_corr2.loc["Mean", :] = value
	df_corr2.loc["Std", :] = std
	df_corr2.to_csv("df_correlation_YieldPredictions_"+ensemble_method+str(ntrees)+"Trees.txt", sep='\t')



	X_for_corr = X.copy()
	X_for_corr["Grain Yield"] = y
	pairwaise_corr = X_for_corr.corr()
	#gene_biomass_corr_sign = np.sign(pairwaise_corr["Biomass"][:-1].values)
	gene_biomass_corr = pairwaise_corr["Grain Yield"][:-1].copy()
	gene_biomass_corr = gene_biomass_corr.reset_index()
	gene_biomass_corr["Sign"] = np.sign(gene_biomass_corr["Grain Yield"])
	gene_biomass_corr.columns = ["Gene", "CorrelationCoeffWYield", "Sign"]
	gene_biomass_corr.to_csv("GenesCorrelationWYield_LabData.txt", sep="\t")


	#print(Feature importance Results for Thousands of Trees Forest
	feature_importances = treeEstimator2.feature_importances_
	B=sorted(range(len(feature_importances)),key=lambda x:feature_importances[x],reverse=True)
	C=sorted(range(len(B)),key=lambda x:B[x])
	df = pd.DataFrame()
	df["Gene Name"] = genes_features_filtered
	df["feat_imp_score"] = feature_importances
	df["rank"] = C
	df["Sign"] = gene_biomass_corr["Sign"]
	#Filter genes with best feature importances on the full lab model
	#indxes_best_features = df[df["rank"]<50].index.values
	df2=df.sort_values(by=['rank'])
	df2.to_csv("feature_importances_"+ensemble_method+str(ntrees)+"Trees.txt", sep='\t')




	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
	print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

	sys.stdout = orig_stdout
	f.close()


if __name__ == '__main__':
    parser = argp.ArgumentParser()
    

    parser.add_argument('-em', '--ensemble_method', type=str, help='Ensemble method: RF for Random Forest or XG for XGBoosting or AB for AdaBoost', default='RF', required=True)
    
    
    args = parser.parse_args()

    main(args.ensemble_method)


