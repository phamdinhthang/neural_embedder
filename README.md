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
