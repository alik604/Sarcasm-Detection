# Scarcasm Detection 
- By [manashpratim](https://github.com/manashpratim/Sarcasm-Detection)
-- Improved by [alik604](https://github.com/alik604/ReadMe)

> Can sarcastic sentences be identified?



# Description
The goal of this project is to detect sarcasm in News Headlines. The dataset that I have used in this project is available for download at  https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection. The dataset contains headlines of news articles, link to the articles and the labels (1: Sarcastic,0: Not Sarcastic). I have used only the headlines to detect sarcasm due to resource constraints. However, I have provided the code to extract the articles from the links, in the notebook. The datset has 26709 headlines. I have split the data into 90:10 training (24038) and test (2671) sets. I have implemented three models namely Convolutional Neural Network (CNN), Bidirectional Gated Recurrent Unit (Bi-GRU) and Bidirectional Long Short Term Memory (Bi-LSTM) using tensorflow and Keras. Aong the models, CNN has the best training accuracy of 99.97% whereas Bi-LSTM has the best validation accuracy of 81.06%. Also, CNN is the most stable among the three models.All the models are trained for 20 epochs.
The accuracies can be further improved by using the articles to detect sarcasm.



# key changes 
## Stacked Bidirectional LSTM
```
## Original 
# acc: 0.9101 | val_acc: 0.8315

## Alik604's 
# acc: 0.8870 | val_acc: 0.83 # CuDNNGRU; speed improvement
# acc: 0.9773 | val_acc: 0.8233 # stacked CuDNNLSTM with dropout
# acc: 0.9267 | val_acc: 0.8311 # stacked CuDNNLSTM with dropout # 5th  epoch,  batch_size=500 , val_loss: 0.4435
  # --arch changed-- 
# acc: 0.9483 | val_acc: 0.8409 # stacked CuDNNLSTM with dropout # 5th  epoch,  batch_size=250 , val_loss: 0.4591
```
## Added 
ConvLSTM2D
`if data_format='channels_first' 5D tensor with shape: (samples, time, channels, rows, cols)`
is a work in progress 

# key learning
This works surprisingly well, despite that it returning sequences
* LSTM
* Dense
* LSTM

*However, may best model did not use this*

```
model_gru = keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(CuDNNLSTM(28, return_sequences=True)),
    Dense(28, activation='relu'), # work well, ive never read about this... 
    Dropout(.3),
    Bidirectional(CuDNNLSTM(20)),

    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

