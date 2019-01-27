# Character Based CNN

PyTorch implementation of a character-level convolutional neural network for text classification

Paper: https://arxiv.org/pdf/1509.01626.pdf

 
![Network architecture](plots/character_cnn.png)

# Why you should care about character level CNNs

- They are quite powerful in text classification (see paper's benchmark) if though they have no notion of semantics
- You don't need to apply any text preprocessing (tokenization, lemmatization, stemming ...) while using them
- They handle misspelled words and OOV (out-of-vocabulary) tokens very well when testing
- They are faster to train compared to recurrent networks
- They are lightweight since they don't require storing a large word embedding matrix, hence you can deploy them in production more easily

# Results

I have tested this model on a set of french labeled customer reviews (over 3 millions). I reported the metrics in TensorboardX. 

The results are petty amazing.

![Training metrics](plots/training_metrics.PNG)





