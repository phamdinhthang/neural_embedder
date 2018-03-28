# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:26:35 2018

@author: ThangPD
"""

import os
from DataLoader import DataLoader
from EntityEmbedder import EntityEmbedder
import json

def main():
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path, label_col = os.path.join(src_path, 'adult_income.csv'), 'high_income'

    data = DataLoader(data_path, label_col)
    print("Numerical columns:",data.numerical_cols)
    print("Categorical columns:",data.categorical_cols)
    print("Categorical columns OHE length:",data.categorical_OHE_length)

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
        embedder = EntityEmbedder(data, layer_nodes=[500, 1000, 1000, 500])
        res = embedder.perform_entity_embedding(learning_rate=0.001, l2_beta=0.01, epochs=10, mini_batch_size=5000)

        fp='C:/Users/admin/Desktop/embedded_result.txt'
        EntityEmbedder.save_to_file(res,fp)

        with open(fp,'r') as f:
            json_content = f.read()
            EntityEmbedder.visualise_embedding_result(json.loads(json_content))

if __name__=='__main__':
    main()