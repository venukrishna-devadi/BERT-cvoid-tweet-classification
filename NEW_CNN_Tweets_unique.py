#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:26:36 2023

@author: venu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:04:38 2023

@author: venu
"""

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import re
import os
import numpy as np

os.getcwd()
os.chdir('/Users/venu/Documents/Productivity/saint peters masters/GA/project/twitterdatacollection (2)/GA_Project_TwitterSentimentAnalysis/Twitter_covid_classification')

raw_data = pd.read_csv('CNN_tweets_unique_.csv')
raw_data.columns

def clean_text(text):
    text = re.sub(r'RT\s@\w+:\s', '', text) # remove retweet text
    text = re.sub(r'https?:\/\/\S+', '', text) # remove URLs
    text = re.sub(r'&\w+;', '', text) # remove HTML entities
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'\W+', ' ', text) # remove special characters
    text = text.lower() # convert to lowercase
    return text


raw_data["clean_text"] = raw_data["text"].apply(clean_text)

## just taking 500 of the rows of raw data just to check how the model works with 500 rows
raw_data_1 = raw_data[["clean_text", 'Labels']][:500]
raw_data_1['Labels'].value_counts()


train_data = raw_data_1.sample(frac = 0.7, random_state = 1994)
validation_data = raw_data_1.drop(train_data.index)

train_data.shape
validation_data.shape

train_data['Labels'].value_counts()
validation_data['Labels'].value_counts()

#train_data_df.columns
train_data = [(text, label) for text, label in zip(train_data['clean_text'], train_data['Labels'])]

# Read validation data from CSV file
validation_data = [(text, label) for text, label in zip(validation_data['clean_text'], validation_data['Labels'])]


# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode the training data
train_texts = [text for text, label in train_data]
train_labels = [label for text, label in train_data]
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)

# Convert encodings and labels to TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(100).batch(16)

# Tokenize and encode the validation data
validation_texts = [text for text, label in validation_data]
validation_labels = [label for text, label in validation_data]
validation_encodings = tokenizer(validation_texts, truncation=True, padding=True, max_length=128)

# Convert encodings and labels to TensorFlow Dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((dict(validation_encodings), validation_labels)).batch(16)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=5, validation_data=validation_dataset)

# Predict on the validation dataset
y_val_true = validation_labels
y_val_pred = model.predict(validation_dataset)
y_val_pred = tf.argmax(y_val_pred.logits, axis=1)

# Calculate evaluation metrics
precision = precision_score(y_val_true, y_val_pred)
recall = recall_score(y_val_true, y_val_pred)
f1 = f1_score(y_val_true, y_val_pred)
accuracy = accuracy_score(y_val_true, y_val_pred)

print(f"Accuracy: {accuracy}")
#96.6%
print(f"F1 score: {f1}")
#91.2%
print(f"Precision: {precision}")
#92.8%
print(f"Recall: {recall}")
#89.6%



unlabelled_data = pd.DataFrame(raw_data["clean_text"][499:])
unlabelled_data.shape
unlabelled_data.columns


# Tokenize and encode the unlabelled data
unlabelled_encodings = tokenizer(unlabelled_data['clean_text'].values.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')
unlabelled_input_ids = unlabelled_encodings['input_ids']
unlabelled_attention_mask = unlabelled_encodings['attention_mask']

# Convert encodings to TensorFlow Dataset
unlabelled_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': unlabelled_input_ids, 'attention_mask': unlabelled_attention_mask})).batch(1)

# Generate predictions for the unlabelled data
unlabelled_predictions = model.predict(unlabelled_dataset)
unlabelled_predicted_labels = tf.argmax(unlabelled_predictions.logits, axis=1)

# Save predicted labels to a CSV file
unlabelled_data['Predicted Labels'] = unlabelled_predicted_labels.numpy()
unlabelled_data.head()
unlabelled_data.columns
unlabelled_data['Predicted Labels'].value_counts()
unlabelled_data.to_csv('CNN_tweets_unique_new_predictions.csv', index=True)







