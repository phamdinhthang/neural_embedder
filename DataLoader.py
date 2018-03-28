# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:15:34 2018

@author: ThangPD
"""
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader(object):
    @staticmethod
    def normalize_data(df):
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
        return df_normalized

    def __init__(self, data_path, label_col):
        self.data_path = data_path
        self.label_col = label_col
        self.embedded_vectors_length = None
        self.read_data()

    def read_data(self):
        df = pd.read_csv(self.data_path)
        df = df.sample(frac=1)
        df = df.reset_index(drop=True)

        #Note: if label col of dataframe is a binary 0-1 cols, it should be encode to True-False, since 0-1 col when convert to one-hot encoding will compose of 1 column OHE only (which softmax loss function is assume to be only one class)
        if set(df[self.label_col].unique()) == set([1,0]):
            df[self.label_col]=df[self.label_col].map({0:'false',1:'true'})

        #Separate of dataframe for label and features
        self.df_label = df.loc[:,[self.label_col]]
        self.df_features = df.drop(self.label_col,axis=1)

        #number of labels one hot index
        self.df_label_OHE = pd.get_dummies(self.df_label)
        self.label_OHE_ncols = len(self.df_label_OHE.columns)

        #Split numerical and categorical variables
        self.numerical_cols = [col for col in self.df_features.columns if is_numeric_dtype(df[col])]
        self.categorical_cols = [col for col in self.df_features.columns if is_string_dtype(df[col])]

        #Get the OHE vector length for each categorical column
        self.categorical_OHE_length = {col:len(self.df_features[col].unique()) for col in self.categorical_cols}

    def set_embedded_length(self,embedded_length):
        if not isinstance(embedded_length,dict):
            print("Invalid embedded length type. Param must of type dict")
            return None
        if embedded_length.keys() != self.categorical_OHE_length.keys():
            print("Invalid embedded length type. Dict keys must equal categorical_OHE_length")
            return None

        for key,val in embedded_length.items():
            if not isinstance(val,int):
                print("Invalid embedded length for key:",key)
                return None

        self.embedded_vectors_length = embedded_length
        return True

    def get_train_test_split_array(self,test_size=0.2):
        #Processing of numerical parts
        self.df_features_numeric = self.df_features.loc[:,self.numerical_cols]
        self.df_features_numeric = DataLoader.normalize_data(self.df_features_numeric)
        self.df_features_numeric_ncols = len(self.numerical_cols)

        #Data frame full, merged from original numerical part and categorical part encoded using One-hot
        self.df_features_full_OHE = self.df_features_numeric

        #A list integer,each integer is the len of OHE vector for corresponding categorical variable
        self.OHE_vectors_length_list = []

        #A list integer,each integer is the len of embedded vector for corresponding categorical variable
        self.embedded_vectors_length_list = []

        #A list of 1D array, each array is the level name of corresponding categorical variable
        self.OHE_labels = []

        #Processing of categorical parts
        for col_name in self.categorical_cols:
            df_col_OHE = pd.get_dummies(self.df_features.loc[:, [col_name]])
            self.df_features_full_OHE = pd.concat([self.df_features_full_OHE, df_col_OHE], axis=1)

            self.OHE_vectors_length_list.append(len(self.df_features[col_name].unique()))
            self.embedded_vectors_length_list.append(self.embedded_vectors_length.get(col_name))
            self.OHE_labels.append(list(df_col_OHE.columns))

        self.total_cols = len(self.df_features_full_OHE.columns)
        self.total_cols_after_embedded = self.df_features_numeric_ncols + sum(self.embedded_vectors_length_list)

        self.features_full_OHE_arr = self.df_features_full_OHE.values
        self.label_OHE_arr = self.df_label_OHE.values

        X_train, X_test, y_train, y_test = train_test_split(self.features_full_OHE_arr, self.label_OHE_arr, test_size=0.2, random_state=42, stratify = self.label_OHE_arr)

        return X_train, X_test, y_train, y_test

    def get_data_batch(self, batch_index, batch_size, X, y):
        data_len = len(X)
        start_idx = batch_index*batch_size
        end_idx = (batch_index+1)*batch_size-1

        if start_idx < data_len and end_idx < data_len:
            return X[start_idx:end_idx,:], y[start_idx:end_idx,:]

        if start_idx < data_len and end_idx >= data_len:
            return X[start_idx:data_len-1,:], y[start_idx:data_len-1,:]

        return None, None


