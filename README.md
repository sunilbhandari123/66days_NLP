# 66days_NLP
# This will be my journey of learning NLP.

# Roadmap


SNo|Blog 1 |Blog 2|
|-|-|-|
|1|[https://blog.futuresmart.ai/nlp-roadmap-2023-step-by-step-guide]|[https://towardsdatascience.com/how-to-get-started-in-nlp-6a62aa4eaeff]

  
# Basics of NLP

SNo| Common NLP Tasks| Approaches to NLP| Challenge in NLP|
|-|-|-|-|
|1| Sentiment analysis|Heuristic methods|More than one meaning of a sentence|
|2| Conversational agents|Machine learning methods|Contextual words|
|3| Knowledge graph and QA systems|Deep learning methods|Colloquialisms, slang and idioms|
|4| Summarization|Deep learning algorithms retain data in sequential order|Tone diff (irony, sarcasm)|
|5| Topic modelling|Auto feature selection|Spelling errors|
|6| Typing behaviour|Creativity in poems, dialogue and script|Diversity of languages|
|7| Text parsing into noun verb|
|8| Speech to text|


# Day 1 

Text Preprocessing with Spacy : Cleaning and Transforming raw text data into a format that can be easily analyzed by machine learning algorithms.Some common task includes Tokenization,Lemmatization,Removing Punctuations.where i learned about the basics of spacy ,sentence boundary detections,token attributes,Part of speech tagging,name Entinty Reconition,word vector and spacy and spacy piplines.

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Untitled.ipynb]


# Day 2

Text Preprocessing with Spacy : I learned how to use spaCy as entityRuler,how to make ruler to correct entity.I also learned how to use the spaCy as Matcher,Attributed token by matcher,Applied matcher,greedy keyword,sorting to its apperence, Adding in sequence and used it to find quotes and speakers from the text.We use spaCy matcher over regex in linguistic components so that lemma of word or identifying if the word is a specific type of an entity.

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Rule%20Base%20Spacy.ipynb]
|2|[https://github.com/sunilbhandari123/66days_NLP/blob/main/SpaCy%20Matcher.ipynb]

# Day 3

Text Preprocessing with Spacy : I learned how to use spaCy custom component in spaCy.If we dont want/ want any instances/label in our pipline the we can do that using custom component we can do this by creatinig a custom pipe that removes all instances from doc.ent we can do it in spaCy easily using custom components.

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Custom%20component%20in%20spaCy.ipynb]

# Day 4

Text Preprocessing with Spacy : I learned how to use Regex in spacy to extract multiword tokens,reconstruct spans,inject the spans into the doc.ents and giving priority to the longer spans using filter spans above is the notebook for that.

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Multiword_token_with_Regex.ipynb]

# Day 5

Text Preprocessing with Spacy : I learned and did some basic text preprocessing in tasks such as lowercasing,removing html tags,removing urls from the text,removing punctutations form the text,chat word/slang word treatment,spelling correction in the text below is the notebook for the same.

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/basic%20text%20preprocessing.ipynb]

# Day 6

Text Representation  :
Today i learned about text representation with label encoding,One hot encoding,Bags of words and Term frequency inverse document frequency(tfidf).label encoding,One hot encoding are not often used in NLP because it doesnot give the silimar representation of the similar words,very high curse of dimensiility,OOV(out of voculabary problem etc.Bags of words where is better than label encoding,One hot encoding because it looks for sentences rather than words and convert it to the vector form yet it have some disadvantages apart from advantages like sparse representation,high curse of dimensiility,Doesnot capture the meaning of the word properly.

Whereas TFIDF is calculated by multiplying the TF=no of repetation words in sentences / no of words in sentences) and IDF=log base e(no of sentences/no of sentences contaning the words) and there fore it also contains the value other than 0 and 1.Its advantages are it is simples and intuatuive,fixed sized-I/p : vocab size,word importance is captured in this there are some disadvantages as well ie sparsity exits,OOV (out of voculabary problem ).

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Bag%20of%20words%20and%20TF-IDF.ipynb]

# Day 7

Text Representation  :

Word2Vec:
Word2vec is a technique in natural language processing for obtaining vector representations of words. These vectors capture information about the meaning of the word based on the surrounding words. The word2vec algorithm estimates these representations by modeling text in a large corpus.

Here the vector is made for the sentences base on the cosine similaruty between the feature representation and vocabulary.
Two different model architectures that can be used by Word2Vec to create the word embeddings are the Continuous Bag of Words (CBOW) model & the Skip-Gram model.

The CBOW architecture comprises a deep learning classification model in which we take in context words as input, X, and try to predict our target word, Y.

For example, if we consider the sentence – “Word2Vec has a deep learning model working in the backend.”, there can be pairs of context words and target (center) words. If we consider a context window size of 2, we will have pairs like ([deep, model], learning), ([model, in], working), ([a, learning), deep) etc. The deep learning model would try to predict these target words based on the context words.

The following steps describe how the model works:
The context words are first passed as an input to an embedding layer (initialized with some random weights) as shown in the Figure below.
The word embeddings are then passed to a lambda layer where we average out the word embeddings.
We then pass these embeddings to a dense SoftMax layer that predicts our target word. We match this with our target word and compute the loss and then we perform backpropagation with each epoch to update the embedding layer in the process.
We can extract out the embeddings of the needed words from our embedding layer, once the training is completed.

In skipgram our input and output will be exchanged as of CBOW

When to use :
 CBOW: use when working with small datasets
 Skipgram: use when working with huge datasets

Advantages:
 Dense matrix
 Semantic info is captured
 OOV problem is solved


SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Word2vec.ipynb]


# Day 8 

Deep Learning:
I covered the basic of NLP Moving for I will need the core understanding of the deep learning to understand the concept of transformers so from today i have started learning the deeplearning and learned about perceptron.Perceptron divides our data into two region  therefore it is known as binary classification and it works only in linear datasets./Sort of linear datasets.

I learned about perceptron technique and the main problem with perceptron technique is that we cannot quantify our result ie we cannot sat that how good is that line performing.And there is the problem of convergence therfore I ended of learning the perceptron loss function. Its indepth mathmetical intuatioins.

SNo| Loss functions| Activation function| Output|
|-|-|-|-|
|1| Hingloss|step|Perceptron|
|2| Log loss Binary cross echo|sigmoid|Logistic Regression(Binary calssification|
|3| Categorical cross entropy|softmax|softmax regression/multiclass classification but output is in probability|
|4| MSE|linear(No activation function)|Linear regression and output is in number|

# Resource 
Campux 100 days of DeepLearning

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Perceptron%20Loss%20Function.ipynb]


# Day 9

Deep Learning:
Today i learned about the multilayer perceptron Notation ie how to donate weight bais and output in the multilayer perceptron ,indepth intuation about the multilayer perceptron.

 General overview:
For solving a non linear data we take two perceptron with different decision boundaries and then superimpose the output of the perceptrons and lastely doing smoothing we will get the requried classification 

Maths:
Calulating the probabilit for the particular element(student) in the two perceptrons and then Adding them and putting them in the sigmoid function to get the new probability.

# How to change Neural Network Architecture

1. Increasing the node in the hidden layers
2. Adding nodes to the input
3. Adding nodes to the output node
4. Increase hidden layer

I also learned about the fordward propragation and how it is calculated mathmetically in the given neural network.

And lastely i used ANN in tenserflow and keras to perfrom CUSTOMER CHURN PREDICTION,HANDWRITTEN CLASSIFICATION,GRADITUTE ADMISSION PREDICTIONS.My aim was not to get good accuracy rather it was how to use ANN for deling with different types of problem as later i am going to learn backpropagation it might be helpful.


# Resource 
Campux 100 days of DeepLearning

SNo|Notebook |
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/getting-started-with-ann.ipynb]
|2|[https://github.com/sunilbhandari123/66days_NLP/blob/main/MNIST_classification_using_ANN.ipynb]
|3|[https://github.com/sunilbhandari123/66days_NLP/blob/main/gradituate-admission-prediction-using-ann.ipynb]


# Day 10

Deep Learning: 
Loss Function Today i learned/revised the concept of the loss function before moving to backpropragation.

Loss Functions is a method of evaluating how well your algorithm is modeling your datasets.
If the loss function value is high then our model is performing poor
If the loss function value is low then our model is performing comparatively well.

# Why loss function is important?
Loss function measures and tells  how well our model is performing and what is our problem in the algorithm so we will get the great decision.Loss function is the eye of  machine learning algorithm.

# Lost function vs Cost function?
loss function is calculated in Single Training example
cost function is calculated in batch of training example.

In the way i learned about Mean Squared error,mean absolute error, huber loss,binary cross entropy,categorical cross entropy, sparse categorical entropy its use case ,which activation function to use ,when to use,its formula ,advantages and disadvantages any many more.

# Resource 
Campux 100 days of DeepLearning


# Day 11

Deep Learning: BackPropagation:
Backpropagation is an  algorithm used to train neural network.

For a given data it will find the optimal values of weights and bais in which our neural network will give us the good results.

1. Initilize the weight and bais (w,b)
2. Select a point (rows)
3. Predict output through fordward propagation (Dot Product)
4. Update weights and bais using Gradient Descent.

Gradient descent 
W(new)=W(old)-learning_rate*dl/dW(old)
B(new)=B(old)-learning_rate*dl/dB(old)

For updating the output from the the fordward propragation  we need tp update the hirarchy . and should use above formula to update it in which second term is derivative of loss with the respect to the weight.

We should calculate the partial derivative of the loss function with the respect to all the trainable parameters.


# Day 12

Deep Learning: BackPropagation:
Today i learned how the backpropragation works how the drevatives is done to all the trainable parameters and why is it important to all the trainable parameters in the neural network and how to impelement the backpropagation by writing own code ,concept of loss function,concept of gradient,Derivative vs Gradient ,concept of derivative ,concept of minima,backpropagation intuation and convergence.

# Resource 
Campux 100 days of DeepLearning

SNo|Notebook|
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Backpropragation.ipynb]


# Day 13

Deep Learning: 
Memozation and its important in backpropagation:

Today i learned about the multilayered percepton Memoization .Memozation is an optimization technique used primarily to speed up computer programs by storing the results of expensive function calls and returning the catched results when the same input occurs again.

Backpropagation = Chain Rule + Memoization


Gradient Descent in Neural Network

1. Batch:
   -We will take the entire dataset and update the value of weight and bais.
   -Faster
   -Total update of weight and bais is equal to the number of epochs
   
2. Stochastic
   -We will taje the single row and update the value of weight and bais
   -Total number of updates of rows and columns are is the multiple of (number of epochs * number of rows)
   -Frequency of weight update is higher than that of the Batch Gradient Descent

  
3. MiniBatch
   - Here we will divide the total number of rows in the batches
   - Then we will update the weights and bais for the one batchs.

Which is faster in term of speed and which converges faster given same number of epochs?
= Given same number of epoches the Batch Gradient Descent is more faster.
= Stochastic Gradient Descent convetges fast due to more number of updates.

Advantage of Stochastic GD is that it helps to move out of local minima.
Disadvantages of Stochastic GD is that it doesnot provides the exact solution it provides the approximate solution.


# Resource 
Campux 100 days of DeepLearning

SNo|Notebook|
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/MLP_Memozation.ipynb]


# Day 14

Deep Learning:
Vanishing Gradient Descent Problem:

Vanishing Gradient Descent problem is encountred when traning Artifical Neural Networks with gradient based learning method( Gradient Descent) and backpropagation.In such method during each iteration of traning each of the neural network weight recived an update proportional to the partial derivatives of the error function with respect to the current function with respect to the current weight.The probel is that in some case the gradient descent will be vanishgly small effectively preventing the weight from changing its value.

In worst case this may completely stop the neural network from futher traning.


Arises in:
Deep Neural Network
Sigmoid/Tanh Activation


How to reconize it?
1. Focous on loss after each epoch there will be no change in loss
2. plot weight graph

How to handel Vanishing Gradient Descent Problem?
1. Reduce Model Complexity
2. Using RelU activation function
3. Proper weight initilization
4. Batch Normalization
5. Using Residual Network


# Resource 
Campux 100 days of DeepLearning

SNo|Notebook|
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Vanishing_Gradient_Descent_Problem.ipynb]


# Day 15

Deep Learning:
How to improve performance of Neural Network:

1.Fine Tuning Neural Network:
 1. Number of hidden layers:
      We can increase our hidden layers until overfitting occurs.Rather than taking one hidden layer with more neurons we can take more hidden layers with less neuron.
    
 2. Number of neurons per layers
      Input layer neuron is equal to the number of feature and output layer is equal to the type of problem we are solving and  for hidden layer there is no specific rule         but it should be more than enough (sufficent).
    
 3. Learning Rate
 4. Optimizer
 5. Batch-size
     There are two approches and it is smaller batch size (8 to 32) and it advantage is generalization will be better and its disadvantage is that it will be slow. Another       approch is large batch size (8192) its advantage is it is faster and its disadvantage is that its generalization will be slow.
    
     We should use large batch size effictively using warning of learning rate ie learning rate scheduler.
    
 6. Activation function

    
 7. Epochs
        we should use more epoch with early stopping .Early stopping is a mechanism which is intelligent enough to understand when to stop..

2. By solving problems
   1. Vanishing / Exploding Gradient
   2. Not enough data
   3. Slow traning
   4. Overfitting



# Day 16 
Deep Learning:

Improving Neural Network Performance

1. Vanishing Gradient
   Activation Functions
   Weight Initalization

2.Overfitting
   Reduce complexity /Increase data
   Dropout layer
   Regularization
   Early stopping

3.Normalization
   Normalizing Input
   Batch Normalization
   Normilizing Activations

4.Gradient checking and clipping
5.Optimizers
   Momentum
   Adagrad
   RMS prep

6.Learning Rate scheduling
7.Hyperparameter Tuning
   No of hidden layers
   Nodes perlayer or Batch_size

Today i learned about the early stopping in Neural Network and why is it important in neural network ,Normalization in Neural Network when the data is not normalized then the loss function will be not symmetrical and when the data is normalized then the loss function will be symmetrical and the solution for it is  Standarization and Normalization.Normalization is used when we know the maximum and the minimum values and standarization is used when we dont know the maximum and the minimum values.Lastely i learned about the dropout layers in the deeplearning and it is used in traning time not in the testing time w=w*(1-p) where p is the dropout layer.

# Resource 
Campux 100 days of DeepLearning

SNo|Notebook|
|-|-|
|1|[https://github.com/sunilbhandari123/66days_NLP/blob/main/Early_Stopping_in_Neural_Network.ipynb]

