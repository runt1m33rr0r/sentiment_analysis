# Sentiment analysis with deep learning

------------------
## Project description
This is a program written in python that does sentiment analysis on user-specified text(either an IMDB review or a tweet). It utlizes deep learning through the keras programming interface included in tensorflow.

------------------
## Requirements
To run this program, you need to have python 3 installed. You also need to install the packages matplotlib, tensorflow or tensorflow-gpu and numpy. You can install them manually using pip or using pip and specifying the requirements.txt file. You can use any operating system that is supported by tensorflow.

------------------
## Used data
The project uses the IMDB data from [here](https://s3.amazonaws.com/text-datasets/aclImdb.zip) and twitter data from [here](https://www.kaggle.com/kazanova/sentiment140).
- To use the data in this project, download both of the datasets and extract them.
- Rename the .csv file from the twitter data from "training.1600000.processed.noemoticon" to "data.csv" and place it inside of the "sentiment_analysis" folder.
- Place the "test" and "train" folders from the IMDB dataset inside of the "data" folder inside "sentiment_analysis".

------------------
## Program usage
#### Running the scripts for the first time might take a while!
After you have installed the program dependencies and downloaded and extracted the data, you can either run the twitter or IMDB script. Both scripts accept a command line argument - the text the user wants to be evaluated.
- For example the IMDB script:
```sh
python3 start_imdb.py "This movie is amazing!"
```
- And the twitter script:
```sh
python3 start_twitter.py "I don't like the new twitter user interface."
```
- You can also run the scripts without specifying some text. This way only the deep learning models will be trained.

Both scripts process the data and create deep learning models. The processed data, deep learning models and data tokens are being saved as files in order to avoid unnecessary calculations when the user runs the program multiple times. To delete the saved program data you should delete the files "data_twitter.pickle", "tokenizer_twitter.pickle", "data_imdb.pickle", "tokenizer_imdb.pickle" and the folders "twitter_nn" and "imdb_nn".

------------------
## Important notes about the data
The data sample taken from both datasets consists of 50 000 elements. The data is divided in 3 portions: training set, validation set and testing set. The training set consists of 15 000 elements, the validation set - 10 000 and the testing set - 25 000. Each element in the set consists of array of words(the review or tweet) and a label that specifies the sentiment of the given text. The sentances are being processed so that only the first 10 000 most frequent words are being taken into account. The sentences are then being one-hot encoded in order to convert them to a compatible data type for the model.

------------------
## Model architecture
The program uses a feedforward neural network with 10 000 inputs and 1 output. It has one hidden layer with 16 inputs and outputs. The output of the network is a single float value between 0 and 1. If the value is closer to 0, the sentiment is negative. If the value is closer to 1, the sentiment is positive. I have used dropout layers and L2 regularization to avoid overfitting as much as possible. The training of the network lasts for 6 epochs with a batch size of 512 elements.
Here is the model in python code:
```python
model = models.Sequential()
model.add(layers.Dense(16,
    kernel_regularizer=regularizers.l2(0.001),
    activation='relu',
    input_shape=(words_count,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(
    16,
    kernel_regularizer=regularizers.l2(0.001),
    activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

------------------
## Model accuracy
After running the program, it shows how accurate the deep learning model is when tested against the test portion of the dataset.
- For the IMDB data, the accuracy is around 88%.
- For the twitter data, the accuracy is around 75%.
