# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:10:20 2018

@author: ThangPD
"""
import tensorflow as tf
from datetime import datetime
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import copy
import json


def tsne_2d_transformed(df, label_col=None):
    df_transformed = copy.deepcopy(df)
    if label_col is not None and label_col in list(df.columns):
        df_transformed = df_transformed.drop('label',axis=1)

    features = df_transformed.values
    n_components = 2
    model = TSNE(n_components=n_components)
    transformed = model.fit_transform(features)

    cols = ['feature_'+str(i) for i in range(n_components)]
    df_transformed =  pd.DataFrame(data=transformed,columns=cols)

    if label_col is not None and label_col in list(df.columns):
        df_transformed['label'] = df['label']
    return df_transformed

def scatter_plot(x,y,label,var_name):
    plt.figure()
    plt.scatter(x, y)
    for i, lbl in enumerate(label):
        plt.annotate(lbl, (x[i], y[i]))
    plt.xlabel('feature_0')
    plt.xlabel('feature_1')
    plt.title("Variable name: "+var_name)
    plt.show()

class EntityEmbedder(object):
    @staticmethod
    def save_to_file(result_dic, filepath):
        try:
            with open(filepath,'w') as f:
                f.write(json.dumps(result_dic))
        except:
            print("Cannot save to:",filepath)

    @staticmethod
    def visualise_embedding_result(result_dic):
        if not isinstance(result_dic,dict): return

        for variable_name, embedded_vectors_dic in result_dic.items():
            df_dic_list = []
            for label_name, embedded_vector in embedded_vectors_dic.items():
                row = {}
                row['label']=label_name
                for index,val in enumerate(embedded_vector):
                    row['features'+str(index)]=val
                df_dic_list.append(row)

            df = pd.DataFrame(df_dic_list)
            df_transformed = tsne_2d_transformed(df, 'label')
            print(df_transformed)

            feature1 = list(df_transformed['feature_0'].values)
            feature2 = list(df_transformed['feature_1'].values)
            label = list(df_transformed['label'].values)

            scatter_plot(feature1,feature2,label,variable_name)
        return

    @staticmethod
    def create_dense_layer(input_tensor, nodes, name, use_relu=True):
        input_shape = input_tensor.get_shape().as_list()

        w = tf.Variable(tf.truncated_normal([input_shape[1], nodes], stddev=1), name=name+'w')
        b = tf.Variable(tf.truncated_normal([nodes], stddev=1), name=name+'b')
        output_tensor = tf.add(tf.matmul(input_tensor, w), b, name = name+'linear_output')
        if use_relu==True:
            output_tensor = tf.nn.relu(output_tensor, name = name+'_relu_output')
        return output_tensor, w

    def __init__(self, data, layer_nodes=[50,100,100,50]):
        #note: data must be instance of DataLoader object. layer_nodes is list of integer: number of neurons of each layer. i.e: [50,100,200,300,50]
        self.data = data
        self.layer_nodes = layer_nodes

    def perform_entity_embedding(self,learning_rate=0.0001, l2_beta=0.01, epochs=20, mini_batch_size=1000):
        X_train, X_test, y_train, y_test = self.data.get_train_test_split_array(test_size=0.2)

        #Place holder for inputs and outputs
        x = tf.placeholder(tf.float32, [None, self.data.total_cols], name = 'x')
        x_shaped = tf.reshape(x, [-1, self.data.total_cols], name = 'x_shaped')
        y = tf.placeholder(tf.float32, [None, self.data.label_OHE_ncols], name = 'y')

        #Slice the input 2D tensor by columns (slice to separate OHE vectors)
        sliced_numeric = tf.slice(x_shaped, begin = [0,0], size = [-1, self.data.df_features_numeric_ncols], name = 'x_shaped_numeric')
        slice_index=self.data.df_features_numeric_ncols
        sliced_OHE_tensors = []
        for index, col_len in enumerate(self.data.OHE_vectors_length_list):
            sliced = tf.slice(x_shaped, begin = [0,slice_index], size = [-1,col_len], name = 'x_shaped_ohe_sliced_' + str(index))
            sliced_OHE_tensors.append(sliced)
            slice_index += col_len

        #Entity Embedding layers
        embedded_weights = []
        output_embeddeds = []
        embedded_weights_index=0
        for sliced, ohe_len, emb_len in zip(sliced_OHE_tensors, self.data.OHE_vectors_length_list, self.data.embedded_vectors_length_list):
            wd = tf.Variable(tf.truncated_normal([ohe_len, emb_len], stddev=2/ohe_len), name='w_ohe_'+str(embedded_weights_index))
            output_embedded = tf.matmul(sliced, wd, name = 'ohe_embedded_'+str(index))
            embedded_weights.append(wd)
            output_embeddeds.append(output_embedded)
            embedded_weights_index+=1

        all_inputs = [sliced_numeric]
        all_inputs.extend(output_embeddeds)
        concatenated_layer = tf.concat(all_inputs, axis=1, name='concatenate')

        #Dense layers
        dense_layer_weights = []
        input_tensor = concatenated_layer
        for index, n_node in enumerate(self.layer_nodes):
            input_tensor, w = EntityEmbedder.create_dense_layer(input_tensor, n_node, 'dense_layer_'+str(index))
            dense_layer_weights.append(w)

        #output layer with softmax activation
        output_layer, _ = EntityEmbedder.create_dense_layer(input_tensor, self.data.label_OHE_ncols, 'output_layer', use_relu=False)

        y_output_probs = tf.nn.softmax(output_layer, name='y_output_probs')
        y_output_labels = tf.argmax(y_output_probs, 1, name = 'y_output_labels')
        y_real_labels = tf.argmax(y, 1, name = 'y_real_labels')

        #Loss function with L2 regularization
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y, name = 'cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')

        l2_weights = [tf.nn.l2_loss(weights) for weights in dense_layer_weights]
        l2_regularizer = tf.add_n([l2_weights], name = 'l2_regularizer')
        l2_loss = tf.reduce_mean(loss + l2_beta*l2_regularizer, name = 'l2_loss')

        #Optimization with Adam on either loss or l2_loss
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l2_loss)

        # define an accuracy assessment operation
        correct_prediction = tf.equal(y_real_labels, y_output_labels, name = 'correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(epochs):
                total_batch = int(len(X_train)/mini_batch_size)
                print("------------------------Epoch:",epoch+1,"/",epochs,"---------------------------")
                for i in range(total_batch):
                    X_mini_batch, y_mini_batch = self.data.get_data_batch(i, mini_batch_size, X_train, y_train)
                    if X_mini_batch is not None and y_mini_batch is not None:
                        _, cost = sess.run([optimiser, l2_loss], feed_dict={x: X_mini_batch, y: y_mini_batch})
                        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: y_train})
                        test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
                        print("-",str(datetime.now().strftime("%Y-%b-%d %H:%M:%S")),", batch:",i+1,'/',total_batch,', Cost={:.3f}'.format(cost),', Accuracy: train={:.3f}'.format(train_accuracy), ', test={:.3f}'.format(test_accuracy))

            #A list of tuple object with structure [(col_name, OHE_labels, embedded_weight)]
            tpl_list = list(zip(self.data.categorical_cols, self.data.OHE_labels, embedded_weights))

            embedding_result = {}
            for tpl in tpl_list:
                embedded_dict = {}
                ohe_labels = tpl[1]
                embedded_weight = tpl[2]
                weight_value = sess.run(embedded_weight)
                col_name = tpl[0]
                for index,label in enumerate(ohe_labels):
                    embedded_vector = weight_value[index,:]
                    #Pandas named OHE colum by 'colname_levelname', subtract the colname from the label
                    embedded_dict[label[len(col_name)+1:]]=embedded_vector.tolist()

                col_name = tpl[0]
                embedding_result[col_name]=embedded_dict

            return embedding_result
