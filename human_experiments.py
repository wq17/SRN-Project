#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 22:16:42 2022

@author: wendyqi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel
from scipy.stats import f_oneway
from scipy.stats import normaltest, kruskal, wilcoxon
from statsmodels.stats import descriptivestats
from statistics import mean, median, stdev
import argparse
from matplotlib.patches import Rectangle
from numpy import corrcoef
from numpy.matlib import repmat

# Function for graphing familiarity ratings from human experiments
def SuperPlot(mat, experiment_name, condition_labels, first_moment='mean'):

	figure, axes = plt.subplots(1)
	dots_left_nudge, dot_spread, box_width = .125, 0.1, .75
	cardinal, gold = (177/255, 21/255, 28/255), (248/255, 194/255, 55/255)
	q75_w, q25_w = np.percentile(mat[:,0], [75,25])
	q75_pw, q25_pw = np.percentile(mat[:,1], [75,25])
	iqr_w = q75_w - q25_w
	iqr_pw = q75_pw - q25_pw

	for condition_index in range(mat.shape[1]):
		# plots standard deviations and measures of central tendency
		#median = np.median(mat[:, condition_index])
		q75, q25 = np.percentile(mat[:,condition_index], [75,25])
		iqr = q75 - q25
		#sd = np.std(mat[:, condition_index])
		left_box_edge = condition_index-box_width/2
		right_box_edge = left_box_edge+box_width
		axes.add_patch(Rectangle((left_box_edge, q25), box_width, iqr, 
                       facecolor=(.6, .6, .6), edgecolor='k'))
		median = np.median(mat[:, condition_index], axis=0)
		plt.plot([left_box_edge, right_box_edge], [median, median], 'k', 
                  linewidth=1.2)

	# Implements equal spacing to make a vertical histogram-like representation
	uniques_list, jitter_counters, max_instance = [], [], 0

	for condition in range(mat.shape[1]):
		unique_scores = np.unique(mat[:, condition])
		uniques_list.append(unique_scores)
		instances = []
		for score in unique_scores:
			instances.append(np.sum(mat[:, condition] == score))
        # Gets max instance for uniform scaling (of distribution heights)
		if np.max(instances) > max_instance: max_instance = np.max(instances)
        # Initializes a jitter counter for each discrete score level
		jitter_counters.append(np.zeros(len(instances))) 

	for subject in range(mat.shape[0]):
		# Plots the actual data points
		jitter = []
		for condition in range(mat.shape[1]):
			jitter_index = np.where(uniques_list[condition] == 
                           mat[subject, condition])
			jitter_count = jitter_counters[condition][jitter_index].squeeze()
			jitter.append(dot_spread*jitter_count/max_instance-dots_left_nudge)
			# Increments the jitter counter for that score
			jitter_counters[condition][jitter_index] += 1

		plt.plot(np.arange(mat.shape[1])+np.array(jitter), mat[subject, :], 
           'o-', markerfacecolor=cardinal, markeredgecolor=gold, 
           color=cardinal)

	plt.ylim(0.5,5.5)
	plt.xlim(-1+box_width/2, mat.shape[1]-box_width/2)
	plt.xticks(np.arange(mat.shape[1]), condition_labels, fontsize=14)
	plt.xlabel('Condition', fontsize=20)
	plt.ylabel('Average Familiarity', fontsize=20)
	plt.title(experiment_name, fontsize=20)
	plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], fontsize=14)
	print(-1+box_width/2, mat.shape[1]-box_width/2)

	words = mat[:,0]
	pw = mat[:,1]
	upper_lim_w = q75_w + 1.5 * iqr_w
	lower_lim_w = q25_w - 1.5 * iqr_w
	upper_lim_pw = q75_pw + 1.5 * iqr_pw
	lower_lim_pw = q25_pw - 1.5 * iqr_pw

	for i in range(len(words)):
		if i > upper_lim_w or i < lower_lim_w:
			np.delete(words, i)
		else:
			continue    

	w_clean = words
	pw_clean = pw

	# Plot whiskers for words boxplot
	plt.plot([left_box_edge - (box_width*5/6), 
              left_box_edge - (box_width*5/6)], [q75_w, max(w_clean)], 
              color='k', linewidth=1.2)
	plt.plot([left_box_edge - (box_width*16/15), 
              left_box_edge - (box_width*3/5)], [max(w_clean), max(w_clean)], 
              color='k', linewidth=1.2)
	
	if min(w_clean) != q25_w:
		plt.plot([left_box_edge - (box_width*5/6), 
                  left_box_edge - (box_width*5/6)], [q25_w, min(w_clean)], 
                  color='k', linewidth=1.2)
		plt.plot([left_box_edge - (box_width*16/15), 
                  left_box_edge - (box_width*3/5)], 
                  [min(w_clean), min(w_clean)], color='k', linewidth=1.2)

	# Plot whiskers for part-words boxplot
	plt.plot([left_box_edge + (box_width/2), left_box_edge + (box_width/2)], 
             [q75_pw, max(pw_clean)], color='k', linewidth=1.2)
	plt.plot([left_box_edge + (box_width/4), left_box_edge + (box_width*3/4)], 
             [max(pw_clean), max(pw_clean)], color='k', linewidth=1.2)
	plt.plot([left_box_edge + (box_width/2), left_box_edge + (box_width/2)], 
             [q25_pw, min(pw_clean)], color='k', linewidth=1.2)
	plt.plot([left_box_edge + (box_width/4), left_box_edge + (box_width*3/4)], 
             [min(pw_clean), min(pw_clean)], color='k', linewidth=1.2)
	plt.show()

############################## MAIN #######################################
# Create empty array for familiarity rating data
df_list = np.empty(shape=[0, 4])
stdev_list = []

# Path of data files for Experiment 1 and Experiment 2
root_path = ["data/experiment_one/","data/experiment_two/"]

if __name__ == '__main__':
    # Read in data from both human experiments
    for index in range(len(root_path)):
        # Experiment 1
        if index == 0:
            for filename in os.listdir(root_path[index]):
                if filename.endswith('.csv'):
            
                    # Read in individual file
                    path = os.path.join(root_path[index], filename)
                    df = pd.read_csv(path, header='infer', 
                                     delim_whitespace = True)
                    names= df.columns.str.split(',').tolist()
                    df= df.iloc[:,0].str.split(',', expand=True)
                    df.columns = names
                    df.dropna(how='all')
                    
                    # Extract stimuli and response information
                    df = df[['stimFile', 'slider_w_1.response_raw']]
                    df = np.array(df, dtype=str)
                    df = df[0:4, :]
                    
                    # Save familiarity ratings and store in array
                    single = np.array([['1.50', '1.50', '1.50', '1.50']])
                    for i in range(len(df)):
                        if df[i, 0] == 'stimuli_Aslin/pabiku_finalx2.wav':
                            single[0,0] = df[i, 1]
                        elif df[i, 0] == 'stimuli_Aslin/tibudo_finalx2.wav':
                            single[0,1] = df[i, 1]
                        elif df[i, 0] == 'stimuli_Aslin/pigola_final.wav':
                            single[0,2] = df[i, 1]
                        elif df[i, 0] == 'stimuli_Aslin/tudaro_final.wav':
                            single[0,3] = df[i, 1]
                            
                    single = np.array(single)
                    single = single.astype(np.float)
                    std = np.std(single)
                    stdev_list.append(std)
            
                    single = np.reshape(single, (1,4))
                    df_list = np.concatenate((df_list, single), axis=0)
                    df_list = df_list.astype(np.float)
                    df_list = np.round(df_list, decimals=3)
                    
                    words = df_list[:,0:2]
                    partwords = df_list[:,2:4]
                    words_av = words.mean(axis=1)
                    partwords_av = partwords.mean(axis=1)
                    data_one = [words_av, partwords_av]
                    data_one = np.column_stack((words_av,partwords_av))
        # Experiment 2             
        elif index == 1:
            df_list = np.empty(shape=[0, 4])
            for filename in os.listdir(root_path[index]):
                if filename.endswith('.csv'):
            
                    # Read in individual file
                    path = os.path.join(root_path[index], filename)
                    cols = pd.read_csv(path, nrows=1).columns
                    df = pd.read_csv(path, usecols=cols)
                    names= df.columns.str.split(',').tolist()
                    df.dropna(how='all')
                    
                    # Extract stimuli and response information
                    df = df[['stimFile', 'slider_w_1.response']]
                    df = np.array(df, dtype=str)
                    df = df[4:8, :]
                    
                    # Save familiarity ratings and store in array
                    single = np.array([['1.50', '1.50', '1.50', '1.50']])
                    for i in range(len(df)):
                        if df[i, 0] == 'stimuli/bogaki.wav':
                            single[0,0] = df[i, 1]
                        elif df[i, 0] == 'stimuli/guloba.wav':
                            single[0,1] = df[i, 1]
                        elif df[i, 0] == 'stimuli/tupidu.wav':
                            single[0,2] = df[i, 1]
                        elif df[i, 0] == 'stimuli/gogebi.wav':
                            single[0,3] = df[i, 1]
                    
                    single = np.array(single)
                    single = single.astype(np.float)
                    std = np.std(single)
                    stdev_list.append(std)
            
                    single = np.reshape(single, (1,4))
                    df_list = np.concatenate((df_list, single), axis=0)
                    df_list = df_list.astype(np.float)
                    df_list = np.round(df_list, decimals=3)
                    
                    words = df_list[:,0:2]
                    partwords = df_list[:,2:4]
                    words_av = words.mean(axis=1)
                    partwords_av = partwords.mean(axis=1)
                    data_two = [words_av, partwords_av]
                    data_two = np.column_stack((words_av,partwords_av))

    # Graph average familiarity ratings for Experiment 1 and Experiment 2
    SuperPlot(data_one, experiment_name="Experiment 1", 
              condition_labels=['words', 'part-words'], first_moment='mean')
    SuperPlot(data_two, experiment_name="Experiment 2", 
              condition_labels=['words', 'part-words'], first_moment='mean')