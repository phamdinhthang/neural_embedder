# Neural Embedder
Tensorflow implementation of Categorical Variable encoder, using Shallow Neural Network entity embedding

### 1. Idea
These implementation takes ideas from this [paper](https://arxiv.org/abs/1604.06737) by Cheng Guo and Felix Berkhahn. The solution reachs 3rd position in Rossman store sale Kaggle [competition](https://www.kaggle.com/c/rossmann-store-sales)

Converting categorical variables to numerical variables is an importance step of the data processing pipeline. Many machine learning model accept only numerical input: logistic regression, support vector machine, neural network. Many machine learning library (scikit-learn, numpy, tensorflow) also require input to be numerical as well.

### 2. Embedder architecture
The architecture of the embedder is simply a shallow neural network with 2 layers: one input and one output. Input layer is the one-hot encoder vector of the categorical variable, output layer is the embedded vector.

![Embedder layer](https://github.com/phamdinhthang/neural_embedder/blob/master/misc/embedding_layer.png "")

Value of the dense weights between input and output layer are trained within a regular neural network classifier. Once trained, the set of weight connect the i<sup>th</sup> neuron of the input layer to the output layer is actually the Embedded vector of the i<sup>th</sup> value of the categorical variable

### 3. Train process
The neural network to train the Embedded weight is similar to a regular neural network, except only for the embedding layer, where dense connection only connect each one-hot encoder vector of each categorical variable to the corressponding embedded layer.

![Train architecture](https://github.com/phamdinhthang/neural_embedder/blob/master/misc/embedder_architecture.png "")

The training process composed of 5 steps
1. Slice the input data to 2 part: numerical variables and categorical variable
2. Perform one-hot encoding of the categorical variable. Set up the length of the embedded vector for each categorical variable. Set up Embedded layer and dense connection
4. Concatenate all the embedded vectors with the numerical variables parts.
5. Train neural network classifier with Gradient Descent.

### 4. Example
Example was done using Cencus Income dataset from UCI [repository](https://archive.ics.uci.edu/ml/datasets/census+income).
The embedded vector length for each categorical variable were set:

```
data.set_embedded_length({'work_class': 4, 'education': 10, 'marital_status': 4, 'occupation': 8, 'relationship': 4, 'race': 3, 'gender': 1, 'native_country': 15})
```
