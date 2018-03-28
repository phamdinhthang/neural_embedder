# Neural Embedder
Tensorflow implementation of Categorical Variable encoder, using Shallow Neural Network entity embedding

### 1. Idea
These implementation takes ideas from this [paper](https://arxiv.org/abs/1604.06737) by Cheng Guo and Felix Berkhahn. The solution reachs 3rd position in Rossman store sale Kaggle [competition](https://www.kaggle.com/c/rossmann-store-sales)

Converting categorical variables to numerical variables is an importance step of the data processing pipeline. Many machine learning model accept only numerical input: logistic regression, support vector machine, neural network. Many machine learning library (scikit-learn, numpy, tensorflow) also require input to be numerical as well.

Different encoding techniques such as one-hot encoding, ordinal encoding, feature hashing...impose different impact on the predictive power of the corresponding categorical variable. Some add some arbitrary information to the variable, some throw

### 3. Embedder architecture

Reference-style: 
![alt text][logo]

[logo]: https://github.com/phamdinhthang/neural_embedder/blob/master/misc/embedder_architecture.png "Logo Title Text 2"
