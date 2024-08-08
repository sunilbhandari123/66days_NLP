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

