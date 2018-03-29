# Neural Embedder
Tensorflow implementation of Categorical Variable encoder, using Shallow Neural Network entity embedding

### 1. Idea
These implementation takes ideas from this [paper](https://arxiv.org/abs/1604.06737) by Cheng Guo and Felix Berkhahn. The solution reachs 3rd position in Rossman store sale Kaggle [competition](https://www.kaggle.com/c/rossmann-store-sales)

Converting categorical variables to numerical variables is an importance step of the data processing pipeline. Many machine learning model accept only numerical input: logistic regression, support vector machine, neural network. Many machine learning library (scikit-learn, numpy, tensorflow) also require input to be numerical as well.

### 2. Embedder architecture
The architecture of the embedder is simply a shallow neural network with 2 layers: one input and one output. Input layer is the one-hot encoder vector of the categorical variable, output layer is the embedded vector. Dense connection between 2 layers composed of weights only. There is no bias unit

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

To run the embedder:
```
python test_neural_embedder.py data_path label_col epochs
```

The three parameters are:
* data_path: path to the .csv dataset file
* label_col: column name of the label
* epochs: number of epochs to run the Gradient Descent optimization

### 5. Result inspection
Result are shown in 2 format: json format for the full embedder result, and plot format for 2-D visualization of the result (using t-SNE dimensionality reduction).

* The json full result (example):
```
{
  "work_class": {
    " Federal-gov": [
      -0.08032485097646713,
      0.16355359554290771,
      -0.04732890799641609,
      -0.04210720211267471
    ],
    " Local-gov": [
      -0.33841240406036377,
      0.130662202835083,
      0.3094033896923065,
      0.005182128865271807
    ],
    " Private": [
      0.00808934960514307,
      -0.10201625525951385,
      -0.3162333071231842,
      -0.32399260997772217
    ],
    " Self-emp-inc": [
      -0.1295398622751236,
      0.23013338446617126,
      0.3397238254547119,
      -0.40874695777893066
    ],
    " Self-emp-not-inc": [
      0.3612673282623291,
      -0.025096414610743523,
      0.03165968880057335,
      0.05519421398639679
    ],
    " State-gov": [
      -0.28594574332237244,
      0.22011640667915344,
      -0.3163900077342987,
      -0.1440461426973343
    ],
    " Without-pay": [
      0.0822419822216034,
      -0.021597225219011307,
      0.44229134917259216,
      0.01329183578491211
    ]
  }
}
```

* The 2-D visualisation result (example):

![Result visualization](https://github.com/phamdinhthang/neural_embedder/blob/master/misc/work_class.png "")

**Note: result and performance may varies depend on the dataset and selected models. **
