## Character Based CNN

[![MIT](https://img.shields.io/badge/license-MIT-5eba00.svg)](https://github.com/ahmedbesbes/character-based-cnn/blob/master/LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/ahmed_besbes_.svg?label=Follow&style=social)](https://twitter.com/ahmed_besbes_)

This repo contains a PyTorch implementation of a character-level convolutional neural network for text classification.

The model architecture comes from this paper: https://arxiv.org/pdf/1509.01626.pdf

![Network architecture](plots/character_cnn.png)

There are two variants: a large and a small. You can switch between the two by changing the configuration file.

This architecture has 6 convolutional layers:

|Layer|Large Feature|Small Feature|Kernel|Pool|
|-|-|-|-|-|
|1|1024|256|7|3|
|2|1024|256|7|3|
|3|1024|256|3|N/A|
|4|1024|256|3|N/A|
|5|1024|256|3|N/A|
|6|1024|256|3|3|

and 2 fully connected layers:

|Layer|Output Units Large|Output Units Small|
|-|-|-|
|7|2048|1024|
|8|2048|1024|
|9|Depends on the problem|Depends on the problem|


## Why you should care about character level CNNs

- They are quite powerful in text classification (see paper's benchmark) even though they don't have any notion of semantics
- You don't need to apply any text preprocessing (tokenization, lemmatization, stemming ...) while using them
- They handle misspelled words and OOV (out-of-vocabulary) tokens
- They are faster to train compared to recurrent neural networks
- They are lightweight since they don't require storing a large word embedding matrix. Hence, you can deploy them in production easily

## Results

I have tested this model on a set of french labeled customer reviews (of over 3 millions rows). I reported the metrics in TensorboardX. 

I got the following results

||F1 score|Accuracy|
|-|-|-|
|train||0.9366|
|test|||

![Training metrics](plots/training_metrics.PNG)

## Dependencies

- numpy 
- pandas
- sklearn
- PyTorch 0.4.1
- tensorboardX
- Tensorflow (to be able to run TensorboardX)

## Structure of the code

At the root of the project, you will have:

- **train.py**: used for training a model
- **predict.py**: used for the testing and inference
- **config.json**: a configuration file for storing model parameters (number of filters, neurons)
- **src**: a folder that contains:
  - **cnn_model.py**: the actual CNN model (model initialization and forward method)
  - **data_loader.py**: the script responsible of passing the data to the training after processing it
  - **utils.py**: a set of utility functions for text preprocessing (url/hashtag/user_mention removal)

## How to use the code

### Training

Launch train.py with the following arguments:

- `data_path`: path of the data. Data should be in csv format with at least a column for text and a column for the label
- `validation_split`: ratio of train data (default to 0.8), the remaining portion is for validation
- `label_column`: column name of the labels
- `text_column`: column name of the texts 
- `max_rows`: the maximum number of rows to load from the dataset. (I mainly use this for testing to go faster)
- `chunksize`: size of the chunks when loading the data using pandas
- `encoding`: default to utf-8
- `steps`: text preprocessing steps to include on the text like hashtag or url removal
- `alphabet`: default to abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~\`+-=<>()[]{} (normally you should not modify it)
- `number_of_characters`: default 68
- `extra_characters`: additional characters that you'd add to the alphabet. For example uppercase letters or accented characters
- `config_path`: this is path of the config json file
- `size`: large / small, depending on the model you choose
- `max_length`: the maximum length to fix for all the documents. default to 150 but should be adapted to your data
- `number_of_classes`: depends on the problem. default to 2 (binary classification problem)
- `epochs`: number of epochs 
- `batch_size`: batch size, default to 128.
- `optimizer`: adam or sgd, default to sgd
- `learning_rate`: default to 0.01
- `schedule`: number of epochs by which the learning rate decreases (learning rate scheduling works only for sgd), default to 3. set it to 0 to disable it
- `patience`: maximum number of epochs to wait without improvement of the validation loss, default to 3
- `checkpoint`: To choose to save the model on disk or not. default to 1, set to 0 to disable model checkpoint
- `workers`: number of workers in PyTorch DataLoader, default to 1
- `log_path`: path of tensorboard log file
- `output`: path of the folder where models are saved
- `model_name`: prefix name of saved models

**Example usage**:

python train.py --data_path=/data/tweets.csv --max_rows=200000 

## License

This project is licensed under the MIT License
