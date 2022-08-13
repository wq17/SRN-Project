#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:07:55 2022

@author: wendyqi
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import statistics
import seaborn as sns
import argparse
from matplotlib.patches import Rectangle
from numpy.matlib import repmat
from tensorflow import keras
from tensorflow.keras import layers
from tabulate import tabulate
from scipy.spatial import distance
from scipy import stats
from scipy.stats import normaltest, wilcoxon
from sklearn import manifold
from calculate_dist import make_sets

########################### Data Processing ##############################

# Read data (word order for training and phoneme featural representation) 
# into dataframes
word_order    = "data/word_order.txt"
mapping_IPA   = "data/mapping.txt"

word_order  = pd.read_csv(word_order, header = None, delim_whitespace = True)
mapping_IPA = pd.read_csv(mapping_IPA,
                          header = None, delim_whitespace = True)

# Convert into array
word_order     = np.array(word_order)
mapping_IPA    = np.array(mapping_IPA)

# Function for making train and test datasets (Phonetic Features simulations)
def make_sets(first_w, second_w, third_w, fourth_w):
    
    # Define vectors for stimuli creation
    word_one = np.zeros((1, 50), dtype = int)
    word_one_o = np.empty((0, 50))
    word_two = np.zeros((1, 50), dtype = int)
    word_two_o = np.empty((0, 50))
    word_three = np.zeros((1, 50), dtype = int)
    word_three_o = np.empty((0, 50))
    word_four = np.zeros((1, 50), dtype = int)
    word_four_o = np.empty((0, 50))
    pw_one = np.zeros((1, 50), dtype = int)
    pw_one_o = np.empty((0, 50))
    pw_two = np.zeros((1, 50), dtype = int)
    pw_two_o = np.empty((0, 50))
    
    pw1 = third_w[4:6] + fourth_w[0:4]
    pw2 = fourth_w[4:6] + third_w[0:4]
    
    # Place input arguments into a list
    all_w = []
    all_w.extend((first_w, second_w, third_w, fourth_w, pw1, pw2))
    
    # Make vector representation of each word
    syl_one = []
    syl_two = []
    syl_three = []
    
    # Make word 1
    for word in range(len(all_w)):
        syl_one = []
        syl_two = []
        syl_three = []
        
        for char in all_w[word][0:2]:
            for i in range(mapping_IPA.shape[0]):
                if char == mapping_IPA[i, 0]:
                    syl_one = np.append(syl_one, mapping_IPA[i, 1:])   
                    
        for char in all_w[word][2:4]:
            for i in range(mapping_IPA.shape[0]):
                if char == mapping_IPA[i, 0]:
                    syl_two = np.append(syl_two, mapping_IPA[i, 1:])
        for char in all_w[word][4:6]:
            for i in range(mapping_IPA.shape[0]):
                if char == mapping_IPA[i, 0]:
                    syl_three = np.append(syl_three, mapping_IPA[i, 1:])
    
        syl_one = np.array(syl_one)
        syl_one = np.reshape(syl_one, (1, 50))
        syl_two = np.array(syl_two)
        syl_two = np.reshape(syl_two, (1, 50))
        syl_three = np.array(syl_three)
        syl_three = np.reshape(syl_three, (1, 50))
    
        if word == 0:
            word_one = np.concatenate((word_one, syl_one, syl_one, syl_two), 
                                       axis=0)
            word_one_o = np.concatenate((word_one_o, syl_two, syl_three), 
                                         axis=0)
        elif word == 1:
            word_two = np.concatenate((word_two, syl_one, syl_one, syl_two), 
                                       axis=0)
            word_two_o = np.concatenate((word_two_o, syl_two, syl_three), 
                                         axis=0)
        elif word == 2:
            word_three = np.concatenate((word_three, syl_one, syl_one, 
                                         syl_two), axis=0)
            word_three_o = np.concatenate((word_three_o, syl_two, syl_three), 
                                           axis=0)
        elif word == 3:
            word_four = np.concatenate((word_four, syl_one, syl_one, syl_two), 
                                        axis=0)
            word_four_o = np.concatenate((word_four_o, syl_two, syl_three), 
                                          axis=0)
        elif word == 4:
            pw_one = np.concatenate((pw_one, syl_one, syl_one, syl_two), 
                                     axis=0)
            pw_one_o = np.concatenate((pw_one_o, syl_two, syl_three), 
                                       axis=0)
        elif word == 5:
            pw_two = np.concatenate((pw_two, syl_one, syl_one, syl_two), 
                                     axis=0)
            pw_two_o = np.concatenate((pw_two_o, syl_two, syl_three), 
                                       axis=0)
 
    # Add up all words (make training familiarization set and test sets)
    words_all = np.empty((0, 50))
    words_all_o = np.empty((0, 50))
    test_all = np.empty((0, 50))
    test_all_o = np.empty((0, 50))
    
    # Create training set (inputs and outputs)
    for i in word_order:
        if i == 1:
            words_all = np.concatenate((words_all, word_one), axis=0)
            words_all_o = np.concatenate((words_all_o, word_one_o), axis=0)
        elif i == 2:
            words_all = np.concatenate((words_all, word_two), axis=0)
            words_all_o = np.concatenate((words_all_o, word_two_o), axis=0)
        elif i == 3:
            words_all = np.concatenate((words_all, word_three), axis=0)
            words_all_o = np.concatenate((words_all_o, word_three_o), axis=0)
        elif i == 4:
            words_all = np.concatenate((words_all, word_four), axis=0)
            words_all_o = np.concatenate((words_all_o, word_four_o), axis=0)
     
    # Create test sets (inputs and outputs)  
    test_all = np.concatenate((test_all, word_one, word_two, pw_one, pw_two))
    test_all_o = np.concatenate((test_all_o, word_one_o, word_two_o, pw_one_o, 
                                 pw_two_o))
    
    return words_all, words_all_o, test_all, test_all_o

# Function for making train and test datasets (MDS)
def make_MDS(first_w, second_w, third_w, fourth_w):
    
    # Function that calculates the distance between the two words
    def calculate_dist(first_syl, second_syl):
        
        syl_one = []
        syl_two = []
        
        for char in first_syl:
            for i in range(mapping_IPA.shape[0]):
                if char == mapping_IPA[i, 0]:
                    syl_one = np.append(syl_one, mapping_IPA[i, 1:])
                    
        for char in second_syl:
            for i in range(mapping_IPA.shape[0]):
                if char == mapping_IPA[i, 0]:
                    syl_two = np.append(syl_two, mapping_IPA[i, 1:])
                    
        dist = distance.euclidean(syl_one, syl_two)
        
        return dist
    
    # Create the list of syllables and matrix for MDS
    syl = [first_w[0:2], first_w[2:4], first_w[4:6], second_w[0:2], 
           second_w[2:4], second_w[4:6], third_w[0:2], third_w[2:4], 
           third_w[4:6], fourth_w[0:2], fourth_w[2:4], fourth_w[4:6]]
    
    matrix = np.zeros(shape=[len(syl), len(syl)])
    
    # Calculate and save distance between each syllable
    for i in range(len(syl)):
        for index in range(i+1, len(syl)):
            dist = calculate_dist(syl[i], syl[index])
            matrix[i, index] = dist
    
    matrix = np.around(matrix, 2)
    transpose = np.transpose(matrix)

    final = matrix + transpose
    
    # Calculate new mappings of syllables through MDS model
    mds_model = manifold.MDS(n_components = 3, dissimilarity = 'precomputed')
    mds_fit = mds_model.fit(final)
    mds_coords = mds_model.fit_transform(final)
    
    data_final = np.array(mds_coords)
    data_min = np.min(mds_coords)
    data_max = np.max(mds_coords)

    # Standardize values
    m = np.mean(data_final)
    s = np.std(data_final)
    
    data_final = (data_final - m)/s
    
    # Normalize values
    data_final = (data_final - data_min) / (data_max - data_min)

    # Make word 1
    word_one = np.zeros((1, 3), dtype = int) # padding
    word_one_o = np.empty((0, 3))
    word_two = np.zeros((1, 3), dtype = int) # padding
    word_two_o = np.empty((0, 3))
    word_three = np.zeros((1, 3), dtype = int)
    word_three_o = np.empty((0, 3))
    word_four = np.zeros((1, 3), dtype = int)
    word_four_o = np.empty((0, 3))
    pw_one = np.zeros((1, 3), dtype = int)
    pw_one_o = np.empty((0, 3))
    pw_two = np.zeros((1, 3), dtype = int)
    pw_two_o = np.empty((0, 3))
    word_full = np.zeros((2, 3), dtype = int)
    zeros = np.zeros((1, 3), dtype = int) # padding
    syl_one = []
    syl_two = []
    syl_three = []
    
    ## Make vector representation of each word
    # Make MDS representation of each word and part-word
    for i in range(1,7):
        syl_one = []
        syl_two = []
        syl_three = []
        
        # Make all syllables of each word and part-word with MDS coordinates
        if i == 1:
            # Make word 1
            syl_one = np.append(syl_one, data_final[0, :])   
            syl_two = np.append(syl_two, data_final[1, :])
            syl_three = np.append(syl_three, data_final[2, :])
        elif i == 2:
            # Make word 2
            syl_one = np.append(syl_one, data_final[3, :])   
            syl_two = np.append(syl_two, data_final[4, :])
            syl_three = np.append(syl_three, data_final[5, :])
        elif i == 3:
            # Make word 3
            syl_one = np.append(syl_one, data_final[6, :])   
            syl_two = np.append(syl_two, data_final[7, :])
            syl_three = np.append(syl_three, data_final[8, :])
        elif i == 4:
            # Make word 4
            syl_one = np.append(syl_one, data_final[9, :])   
            syl_two = np.append(syl_two, data_final[10, :])
            syl_three = np.append(syl_three, data_final[11, :])
        elif i == 5:
            # Make part-word 1
            syl_one = np.append(syl_one, data_final[8, :])   
            syl_two = np.append(syl_two, data_final[9, :])
            syl_three = np.append(syl_three, data_final[10, :])
        elif i == 6:
            # Make part-word 2
            syl_one = np.append(syl_one, data_final[11, :])   
            syl_two = np.append(syl_two, data_final[6, :])
            syl_three = np.append(syl_three, data_final[7, :])
            
        syl_one = np.array(syl_one)
        syl_one = np.reshape(syl_one, (1, 3))
        syl_two = np.array(syl_two)
        syl_two = np.reshape(syl_two, (1, 3))
        syl_three = np.array(syl_three)
        syl_three = np.reshape(syl_three, (1, 3))
        
        # Combine syllables to make input & output values for each word and 
        # part-word
        if i == 1:
            word_one = np.concatenate((word_one, syl_one, syl_one, syl_two), 
                                       axis=0)
            word_one_o = np.concatenate((word_one_o, syl_two, syl_three), 
                                         axis=0)
        elif i == 2:
            word_two = np.concatenate((word_two, syl_one, syl_one, syl_two), 
                                       axis=0)
            word_two_o = np.concatenate((word_two_o, syl_two, syl_three), 
                                         axis=0)
        elif i == 3:
            word_three = np.concatenate((word_three, syl_one, syl_one, 
                                         syl_two), axis=0)
            word_three_o = np.concatenate((word_three_o, syl_two, syl_three), 
                                           axis=0)
        elif i == 4:
            word_four = np.concatenate((word_four, syl_one, syl_one, syl_two), 
                                        axis=0)
            word_four_o = np.concatenate((word_four_o, syl_two, syl_three), 
                                          axis=0)
        elif i == 5:
            pw_one = np.concatenate((pw_one, syl_one, syl_one, syl_two), 
                                     axis=0)
            pw_one_o = np.concatenate((pw_one_o, syl_two, syl_three), 
                                       axis=0)
        elif i == 6:
            pw_two = np.concatenate((pw_two, syl_one, syl_one, syl_two), 
                                     axis=0)
            pw_two_o = np.concatenate((pw_two_o, syl_two, syl_three), 
                                       axis=0)

    ## Add up all words (make training set and test sets)
    words_all = np.empty((0, 3))
    words_all_o = np.empty((0, 3))
    test_all = np.empty((0, 3))
    test_all_o = np.empty((0, 3))
    
    # Make training set (inputs and outputs)
    for i in word_order:
        if i == 1:
            words_all = np.concatenate((words_all, word_one), axis=0)
            words_all_o = np.concatenate((words_all_o, word_one_o), axis=0)
        elif i == 2:
            words_all = np.concatenate((words_all, word_two), axis=0)
            words_all_o = np.concatenate((words_all_o, word_two_o), axis=0)
        elif i == 3:
            words_all = np.concatenate((words_all, word_three), axis=0)
            words_all_o = np.concatenate((words_all_o, word_three_o), axis=0)
        elif i == 4:
            words_all = np.concatenate((words_all, word_four), axis=0)
            words_all_o = np.concatenate((words_all_o, word_four_o), axis=0)
     
    # Make test set (inputs and outputs)
    test_all = np.concatenate((test_all, word_one, word_two, pw_one, pw_two))
    test_all_o = np.concatenate((test_all_o, word_one_o, word_two_o, pw_one_o, 
                                 pw_two_o))

    return words_all, words_all_o, test_all, test_all_o

# Function for building, compiling, training and testing model (MDS)
def model_mds(train_in, train_out, test_in, test_out):
    
    # Model architecture
    model = keras.Sequential()
    model.add(layers.SimpleRNN(units=6, input_shape=[2, 3], 
                               activation="sigmoid"))
    model.add(layers.Dense(units=3, activation='linear'))
    model.summary()
    
    # Compile the model and save weights
    model.compile(loss='mean_squared_error', optimizer='Adam', 
                  metrics=['accuracy'])
    model.save_weights('init_model.h5')
    
    words_list = []
    partwords_list = []
    
    ## Train model and look at loss values for test items 
    # (words and part-words)
    for i in range (70):
        # Train model
        loss = []
        model.load_weights('init_model.h5') # Reload initial weights
        model.fit(train_in, train_out, batch_size=9, epochs=5, verbose=0)
        
        # Test model and extract loss values for individual syllables
        for i in range(len(test_in)):
            results = model.evaluate(test_in[[i],:,:], test_out[[i]], 
                                     verbose=0)
            loss.append(results[0])
        
        # Extract average loss values for words and part-words
        words = loss[0:4]
        pw = loss[4:8]
        words_av = statistics.mean(words)
        partwords_av = statistics.mean(pw)
    
        words_list.append(words_av)
        partwords_list.append(partwords_av)
        data = [words_list, partwords_list]
    
    data = [words_list, partwords_list]
    data = np.column_stack((words_list,partwords_list))
    return data

# Function for compiling, training and testing model (phonetic features)
def model(train_in, train_out, test_in, test_out):
    
    # Model architecture
    model = keras.Sequential()
    model.add(layers.SimpleRNN(units=8, input_shape=[2, 50], 
                               activation="relu"))
    model.add(layers.Dense(units=50, activation='sigmoid'))
    model.summary()

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='Adam', 
                  metrics=['accuracy']) 
    model.save_weights('init.h5')

    words_list = []
    partwords_list = []
    
    ## Train model and extract loss values for test items (words & part-words)
    for i in range (70):
        # Train model
        loss = []
        model.load_weights('init.h5')
        model.fit(train_in, train_out, batch_size=9, epochs=5, verbose=0)
        
        # Test model and extract loss values for individual syllables
        for i in range(len(test_in)):
            results = model.evaluate(test_in[[i],:,:], test_out[[i]], 
                                     verbose=0)
            loss.append(results[0])
        
        # Extract average loss values for words and part-words
        words = loss[0:4]
        pw = loss[4:8]
        words_av = statistics.mean(words)
        partwords_av = statistics.mean(pw)
        
        words_list.append(words_av)
        partwords_list.append(partwords_av)
        
    data = [words_list, partwords_list]
    data = np.column_stack((words_list,partwords_list))
    return data

# Function for graphing loss values from model output
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
		q75, q25 = np.percentile(mat[:,condition_index], [75,25])
		iqr = q75 - q25
		left_box_edge = condition_index-box_width/2
		right_box_edge = left_box_edge+box_width
		axes.add_patch(Rectangle((left_box_edge, q25), box_width, iqr, 
                       facecolor=(.6, .6, .6), edgecolor='k'))
		median = np.median(mat[:, condition_index], axis=0)
		plt.plot([left_box_edge, right_box_edge], [median, median], 'k', 
                  linewidth=1.2)

	# gets them equally spaced to make a vertical histogram-like representation
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
		# this plots the actual data points
		jitter = []
		for condition in range(mat.shape[1]):
			jitter_index = np.where(uniques_list[condition] == mat[subject, 
                                    condition])
			jitter_count = jitter_counters[condition][jitter_index].squeeze()
			jitter.append(dot_spread*jitter_count/max_instance-dots_left_nudge)
            # Increments the jitter counter for that score
			jitter_counters[condition][jitter_index] += 1

		plt.plot(np.arange(mat.shape[1])+np.array(jitter), mat[subject, :], 
                           'o-', markerfacecolor=cardinal, 
                           markeredgecolor=gold, color=cardinal)

	plt.ylim(0, 0.4)
	plt.xlim(-1+box_width/2, mat.shape[1]-box_width/2)
	plt.xticks(np.arange(mat.shape[1]), condition_labels, fontsize=14)
	plt.xlabel('Condition', fontsize=20)
	plt.ylabel('Loss', fontsize=20)
	plt.title(experiment_name, fontsize=20)
	plt.yticks([0, .1, .2, .3, .4], fontsize=14)

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

def DoPlot():
	parser = argparse.ArgumentParser(description='Plot condition-wise \
                                     comparisons of data distributions that \
                                     include 4+ informative pieces of visual \
                                     information about the data')

	#MANDATORIES (ARGS)
	parser.add_argument('mat_fname', 
                        help='path to .npy file containing matrix to be \
                        plotted')
	parser.add_argument('experiment_name', 
                        help='the name of the experiment being plotted')
	parser.add_argument('condition_labels', nargs='+', 
                        help="the condition labels (corresponding to the \
                        matrix's columns) to place below each condition's \
                        distribution")

	#OPTIONALS (KWARGS)
	parser.add_argument('--first-moment', dest='first_moment', default='both', 
                        help='measure of central tendency to plot for each \
                        distribution (options: "mean", "median", or "both")')

	args = parser.parse_args()

	mat = np.load(args.mat_fname)
	assert(len(args.condition_labels) == mat.shape[1])

	SuperPlot(mat, args.experiment_name, args.condition_labels, 
              args.first_moment)

############################## MAIN #######################################
if __name__ == '__main__':
    # Make and reshape data arrays (both phonetic features and MDS)
    train_in, train_out, test_in, test_out = make_sets('pabiku', 'tibudo', 
                                                       'golatu', 'daropi')
    train_in  = np.reshape(train_in, [540,2,train_in.shape[1]])
    test_in  = np.reshape(test_in, [-1,2,test_in.shape[1]])

    train_in_sim2, train_out_sim2, \
    test_in_sim2, test_out_sim2 = make_sets('guloba', 'bogaki', 
                                            'gebitu', 'pidugo')
    train_in_sim2  = np.reshape(train_in_sim2, [540,2,train_in_sim2.shape[1]])
    test_in_sim2  = np.reshape(test_in_sim2, [-1,2,test_in_sim2.shape[1]])

    train_in_mds, train_out_mds, \
    test_in_mds, test_out_mds = make_MDS('pabiku', 'tibudo', 
                                         'golatu', 'daropi')
    train_in_mds = np.reshape(train_in_mds, [540,2,train_in_mds.shape[1]])
    test_in_mds = np.reshape(test_in_mds, [-1,2,test_in_mds.shape[1]])
    
    train_in_mds_sim2, train_out_mds_sim2, \
    test_in_mds_sim2, test_out_mds_sim2 = make_MDS('guloba', 'bogaki', 
                                                   'gebitu', 'pidugo')
    train_in_mds_sim2 = np.reshape(train_in_mds_sim2, 
                                   [540,2,train_in_mds_sim2.shape[1]])
    test_in_mds_sim2 = np.reshape(test_in_mds_sim2, 
                                  [-1,2,test_in_mds_sim2.shape[1]])
    
    # Train and test the model
    data = model(train_in, train_out, test_in, test_out)
    data_sim2 = model(train_in_sim2, train_out_sim2, 
                      test_in_sim2, test_out_sim2)
    data_mds = model_mds(train_in_mds, train_out_mds, 
                         test_in_mds, test_out_mds)
    data_mds_sim2 = model_mds(train_in_mds_sim2, train_out_mds_sim2, 
                              test_in_mds_sim2, test_out_mds_sim2)
    
    # Visualize the loss values
    graph = SuperPlot(data, experiment_name='Simulation 1 (Phonetic Features)', 
                      condition_labels=['words', 'part-words'], 
                      first_moment='mean')
    graph2 = SuperPlot(data_sim2, 
                       experiment_name='Simulation 2 (Phonetic Features)', 
                       condition_labels=['words', 'part-words'], 
                       first_moment='mean')
    graph_mds = SuperPlot(data_mds, experiment_name='Simulation 1 (MDS)', 
                          condition_labels=['words', 'part-words'], 
                          first_moment='mean')
    graph_mds_sim2 = SuperPlot(data_mds_sim2, 
                               experiment_name='Simulation 2 (MDS)', 
                               condition_labels=['words', 'part-words'], 
                               first_moment='mean')
    
    # Wilcoxon signed-rank test (matched pairs) for Experiment 1 and 2
    wilcox_one = wilcoxon(x=data[:,0], y=data[:,1], zero_method='wilcox', 
                          correction=False, alternative='two-sided', 
                          mode='auto')
    wilcox_two = wilcoxon(x=data_sim2[:,0], y=data_sim2[:,1], 
                          zero_method='wilcox', correction=False, 
                          alternative='two-sided', mode='auto')
    wilcox_one_mds = wilcoxon(x=data_mds[:,0], y=data_mds[:,1], 
                              zero_method='wilcox', correction=False, 
                              alternative='two-sided', mode='auto')
    wilcox_two_mds = wilcoxon(x=data_mds_sim2[:,0], y=data_mds_sim2[:,1], 
                              zero_method='wilcox', correction=False, 
                              alternative='two-sided', mode='auto')
    