#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:58:54 2023

@author: wendyqi
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import statistics
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from scipy.spatial import distance
from scipy import stats
from scipy.stats import normaltest, wilcoxon

mapping_IPA = "directory/mapping.txt"
mapping_IPA = pd.read_csv(mapping_IPA, header = None, delim_whitespace = True)
mapping_IPA = np.array(mapping_IPA)

languages = "directory/languages.txt"
languages  = pd.read_csv(languages, header = None, delim_whitespace = True)
languages  = np.array(languages)

word_order = "directory/saffran_vector.txt"
word_order = pd.read_csv(word_order, header = None, delim_whitespace = True)
word_order = np.array(word_order)

language_data = []
array_final = np.zeros(shape=[33, 5])


# Function that returns syllable and word dissimilarity average
def calculate_dist_av(first_w, second_w, third_w, fourth_w):
    
    syl_one_1, syl_two_1, syl_three_1, syl_one_2, syl_two_2, syl_three_2, \
        syl_one_3, syl_two_3, syl_three_3, syl_one_4, syl_two_4, syl_three_4 \
            = ([] for i in range(12))
    word_one = []
    word_two = []
    word_three = []
    word_four = []
    word_dist = []
    
    # Mean syllable dissimilarity
    for char in first_w:
        for i in range(mapping_IPA.shape[0]):
            if char in first_w[0:2] and char == mapping_IPA[i, 0]:
                syl_one_1 = np.append(syl_one_1, mapping_IPA[i, 1:])
            elif char in first_w[2:4] and char == mapping_IPA[i, 0]:
                syl_two_1 = np.append(syl_two_1, mapping_IPA[i, 1:])
            elif char in first_w[4:6] and char == mapping_IPA[i, 0]:
                syl_three_1 = np.append(syl_three_1, mapping_IPA[i, 1:])
                
    for char in second_w:
        for i in range(mapping_IPA.shape[0]):
            if char in second_w[0:2] and char == mapping_IPA[i, 0]:
                syl_one_2 = np.append(syl_one_2, mapping_IPA[i, 1:])
            elif char in second_w[2:4] and char == mapping_IPA[i, 0]:
                syl_two_2 = np.append(syl_two_2, mapping_IPA[i, 1:])
            elif char in second_w[4:6] and char == mapping_IPA[i, 0]:
                syl_three_2 = np.append(syl_three_2, mapping_IPA[i, 1:])
                
    for char in third_w:
        for i in range(mapping_IPA.shape[0]):
            if char in third_w[0:2] and char == mapping_IPA[i, 0]:
                syl_one_3 = np.append(syl_one_3, mapping_IPA[i, 1:])
            elif char in third_w[2:4] and char == mapping_IPA[i, 0]:
                syl_two_3 = np.append(syl_two_3, mapping_IPA[i, 1:])
            elif char in third_w[4:6] and char == mapping_IPA[i, 0]:
                syl_three_3 = np.append(syl_three_3, mapping_IPA[i, 1:]) 
    
    for char in fourth_w:
        for i in range(mapping_IPA.shape[0]):
            if char in fourth_w[0:2] and char == mapping_IPA[i, 0]:
                syl_one_4 = np.append(syl_one_4, mapping_IPA[i, 1:])
            elif char in fourth_w[2:4] and char == mapping_IPA[i, 0]:
                syl_two_4 = np.append(syl_two_4, mapping_IPA[i, 1:])
            elif char in fourth_w[4:6] and char == mapping_IPA[i, 0]:
                syl_three_4 = np.append(syl_three_4, mapping_IPA[i, 1:])

    
    # Mean word dissimmilarity
    for char in first_w:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                word_one = np.append(word_one, mapping_IPA[i, 1:])
                
    for char in second_w:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                word_two = np.append(word_two, mapping_IPA[i, 1:])
             
    for char in third_w:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                word_three = np.append(word_three, mapping_IPA[i, 1:])
    
    for char in fourth_w:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                word_four = np.append(word_four, mapping_IPA[i, 1:])
              
    words = [word_one, word_two, word_three, word_four]
    
    for i in range(len(words)):
        for index in range(i+1, len(words)):
            dist_w = distance.euclidean(words[i], words[index])
            word_dist.append(dist_w)
            
    word_av = statistics.mean(word_dist)
    # print("Mean word dissimilarity: " + str(word_av))
    
    # Mean word/pw dissimilarity
    # Make part-words
    pw1 = third_w[4:6] + fourth_w[0:4]
    pw2 = fourth_w[4:6] + third_w[0:4]
    pw_one = []
    pw_two = []
    pw_dist = []
    
    for char in pw1:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                pw_one = np.append(pw_one, mapping_IPA[i, 1:])
                
    for char in pw2:
        for i in range(mapping_IPA.shape[0]):
            if char == mapping_IPA[i, 0]:
                pw_two = np.append(pw_two, mapping_IPA[i, 1:])
    
    words_pw = [word_one, word_two, pw_one, pw_two]
    
    for i in range(len(words_pw)):
        for index in range(i+1, len(words_pw)):
            dist_pw = distance.euclidean(words_pw[i], words_pw[index])
            pw_dist.append(dist_pw)
            
    pw_av = statistics.mean(pw_dist)
    # print("Mean word/pw dissimilarity: " + str(pw_av))
    
    return word_av, pw_av

# Run model and return mean value
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
        model.fit(train_in, train_out, batch_size=9, epochs=5, verbose=0, shuffle=False)
        
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
    mean_words = statistics.mean(data[0])
    mean_pw = statistics.mean(data[1])
    data = np.column_stack((words_list,partwords_list))
    return mean_words, mean_pw

# For each language, get word-word similarity, word-partword similarity, 
# mean word loss & mean partword loss
def make_chart(languages):
    # For every set of languages
    for i in range(1, 33):
        first_w = np.char.add(languages[i,0], languages[i,1])
        first_w = np.char.add(first_w, languages[i,2])
        first_w  = str(first_w)
        
        second_w= np.char.add(languages[i,3], languages[i,4])
        second_w = np.char.add(second_w, languages[i,5])
        second_w  = str(second_w)
        
        third_w = np.char.add(languages[i,6], languages[i,7])
        third_w = np.char.add(third_w, languages[i,8])
        third_w  = str(third_w)
        
        fourth_w = np.char.add(languages[i,9], languages[i,10])
        fourth_w = np.char.add(fourth_w, languages[i,11])
        fourth_w  = str(fourth_w)
        
        array_final[i,0] = i
        word_sim, pw_sim = calculate_dist_av(first_w, second_w, third_w, fourth_w)
        array_final[i,1] = word_sim
        array_final[i,2] = pw_sim
        
        train_in, train_out, test_in, test_out = make_sets(first_w, second_w, 
                                                           third_w, fourth_w)
        train_in  = np.reshape(train_in, [540,2,train_in.shape[1]])
        test_in  = np.reshape(test_in, [-1,2,test_in.shape[1]])
        
        mean_words, mean_pw = model(train_in, train_out, test_in, test_out)
        array_final[i,3] = mean_words
        array_final[i,4] = mean_pw
        
    # Export chart as .csv file
    chart = {'Language': array_final[:, 0], 
             'Word-Word similarity': array_final[:, 1], 
             'Word-Partword similarity': array_final[:, 2], 
             'Word loss (av)': array_final[:, 3], 
             'Partword loss (av)': array_final[:, 4]}
    df = pd.DataFrame(data=chart)
    df.to_csv('directory/languages.csv')
        
    return array_final

################################## Main #########################################
if __name__== '__main__':
    
    # Make chart
    make_chart(languages)
    print(array_final)
