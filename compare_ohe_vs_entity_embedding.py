# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:18:11 2018

@author: ThangPD
"""
import os
import copy
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def normalize_data(df):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
    return df_normalized

def test_model(model,model_name,X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print(model_name,"accuracy = {:.2f}".format(accuracy_score(y_test,preds)))

def one_hot_encode_test(df,label_col,models):
    df_features = df.drop(label_col,axis=1)
    df_features_ohe = pd.get_dummies(df_features)
    df_features_ohe = normalize_data(df_features_ohe)

    X = df_features_ohe.values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12,stratify=y)
    for model_name,model in models.items():
        test_model(model,model_name,X_train, X_test, y_train, y_test)

def entity_embedding_test(df,label_col,models,embedding_dict_path):
    with open(embedding_dict_path,'r') as f:
        json_content = f.read()
        embedded_dic = json.loads(json_content)
        categorical_cols = list(embedded_dic.keys())

    df_features = df.drop(label_col,axis=1)
    df_features_dic_list = df_features.to_dict('records')
    embedded_features_dic_list = []
    for row_dict in df_features_dic_list:
        row_dict_embedded = copy.deepcopy(row_dict)
        for key,val in row_dict.items():
            if key in categorical_cols:
                embedded_vector = embedded_dic.get(key).get(val)
                for i in range(len(embedded_vector)):
                    row_dict_embedded[key+'_embedded_'+str(i)]=embedded_vector[i]
                row_dict_embedded.pop(key, None)
        embedded_features_dic_list.append(row_dict_embedded)

    embedded_features_df = pd.DataFrame(embedded_features_dic_list)
    embedded_features_df = normalize_data(embedded_features_df)

    X = embedded_features_df.values
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=12,stratify=y)
    for model_name,model in models.items():
        test_model(model,model_name,X_train, X_test, y_train, y_test)

def main():
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path, label_col = os.path.join(src_path, 'adult_income.csv'), 'high_income'
    df = pd.read_csv(data_path)

    models = {'Logistic Regression':LogisticRegression(),
              'Random Forest':RandomForestClassifier(),
              'MLP':MLPClassifier()}

    print("-------------One hot encode--------------")
    one_hot_encode_test(df,label_col,models)
    print("-------------Entity embedding------------")
    embedding_dict_path = 'C:/Users/admin/Desktop/embedded_result.txt'
    entity_embedding_test(df,label_col,models,embedding_dict_path)

if __name__=='__main__':
    main()
