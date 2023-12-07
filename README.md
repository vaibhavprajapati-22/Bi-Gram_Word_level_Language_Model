# Bi-Gram Word level Language Model Using Pytorch
This repository contains the code for a Bi-Gram Word Level Language Model. The model is designed to predict the next word in a sequence based on the previous word, utilizing a bi-gram approach.

### Overview
The language model is implemented in Python and can be run using Google Colab. It leverages the power of bi-gram analysis to understand the probabilistic relationships between consecutive words in a given text.

The bigram model approximates the probability of a word given all the previous words P(wn|w1:n−1) by using only the conditional probability of the preceding word P(wn|wn−1).  In other words, instead of computing the probability P(wn|w1:n−1) we approximate it with the probability P(wn|wn−1). When we use a bigram model to predict the conditional probability of the next word, we are thus making the following approximation: 
P(wn|w1:n−1) ≈ P(wn|wn−1). The assumption that the probability of a word depends only on the previous word is Markov called a Markov assumption. Markov models are the class of probabilistic models that assume we can predict the probability of some future unit without looking too far into the past.

The jupyter notebook contains two models BiGramLanguageModel and BiGramLanguageModel_. Both models have same number of parameter i.e. vocab_size * vocab_size. Actually the layer W or C is the 
vocab_size * vocab_size matrix that contain the log(counts). In log(counts), counts is the number of times a particular word appears after a given word. 

## How to run
1. Save the pride_and_prejudice.txt and BiGram_Word_level_Language_Model file on your local computer or upload the files in Google colab.(I would prefer to upload it in google colab to get GPU acceleration)
2. Run the notebook cells by cell
   Note : Cell containing the below lines need to be run only once
          import nltk
          nltk.download('punkt')
3. On the last cell sampling code is written which generatees 5 sentences

References :
   1. https://web.stanford.edu/~jurafsky/slp3/3.pdf
   2. https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3
      


        
