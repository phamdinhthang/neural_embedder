# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:26:35 2018

@author: ThangPD
"""

import os
from DataLoader import DataLoader
from NeuralEmbedder import NeuralEmbedder
import json
import sys

def neural_embed(data_path, label_col, epochs):
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path, label_col = os.path.join(src_path, 'adult_income.csv'), 'high_income'

    data = DataLoader(data_path, label_col)
    print("Numerical variables:",data.numerical_cols)
    print("Categorical variables:",data.categorical_cols)
    print("Categorical variables OHE length:",data.categorical_OHE_length)

    #Define an embedded vector length for every categorical variable. This is the hyper-parameter that needs to be tuned
    valid = data.set_embedded_length({'work_class': 4,
                                      'education': 10,
                                      'marital_status': 4,
                                      'occupation': 8,
                                      'relationship': 4,
                                      'race': 3,
                                      'gender': 1,
                                      'native_country': 15})

    if valid==True:
        embedder = NeuralEmbedder(data, layer_nodes=[100, 200, 200, 100])
        res = embedder.perform_neural_embedding(learning_rate=0.001, l2_beta=0.01, epochs=10, mini_batch_size=1000)

        res_path='C:/Users/admin/Desktop/embedded_result.txt'
        NeuralEmbedder.save_to_file(res,res_path)

        with open(res_path,'r') as f:
            json_content = f.read()
            NeuralEmbedder.visualise_embedding_result(json.loads(json_content))

if __name__=='__main__':
#    data_path = sys.argv[1]
#    label_col = sys.argv[2]
#    epochs = sys.argv[3]
#    neural_embed(data_path, label_col, epochs)
    neural_embed(None, None, None)