
import json
import nltk
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np

from torch import nn, optim
import torch.nn.functional as F

from collections import Counter



file = ''

with open(file, 'r') as f:
    twits = json.load(f)

print(twits['data'][:10])
print('Number of tweets: {}'.format(len(twits['data'])))


messages = [twit['message_body'] for twit in twits['data']]
# Since the sentiment scores are discrete, we'll scale the sentiments to 0 to 4 for use in our network
sentiments = [twit['sentiment'] + 2 for twit in twits['data']]

nltk.download('wordnet')
import re

def preprocess(message):
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    text = re.sub(r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', ' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub(r'[$]\w+', ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub(r'[@]\w+', ' ', text)

    # Replace everything not a letter with a space
    text = re.sub(r'[\W_]+', ' ', text)

    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(i) for i in tokens if len(i) > 1]
    
    return tokens

tokenized = [preprocess(i['message_body']) for i in twits['data']]

bow = Counter([item for l in tokenized for item in l])

#Frequency of words appearing in message
freqs = {k: v / len(tokenized) for k, v in dict(bow).items()}
# Float that is the frequency cutoff. Drop words with a frequency that is lower or equal to this number.
low_cutoff = 1e-5
# Integer that is the cut off for most common words. Drop words that are the `high_cutoff` most common words.
high_cutoff = 18
# The k most common words in the corpus. Use `high_cutoff` as the k.
K_most_common = bow.most_common(high_cutoff)

filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]
print(K_most_common)
len(filtered_words) 

# A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word. 
vocab = {word: i for i, word in enumerate(filtered_words, 1)}
# Reverse of the `vocab` dictionary. The key is word id and value is the word. 
id2vocab = {word: i for i, word in vocab.items()}
# tokenized with the words not in `filtered_words` removed.
filtered = [[word for word in message if word in vocab] for message in tqdm(tokenized)]

balanced = {'messages': [], 'sentiments':[]}

n_neutral = sum(1 for each in sentiments if each == 2)
N_examples = len(sentiments)
keep_prob = (N_examples - n_neutral)/4/n_neutral

for idx, sentiment in enumerate(sentiments):
    message = filtered[idx]
    if len(message) == 0:
        # skip this message because it has length zero
        continue
    elif sentiment != 2 or random.random() < keep_prob:
        balanced['messages'].append(message)
        balanced['sentiments'].append(sentiment) 

n_neutral = sum(1 for each in balanced['sentiments'] if each == 2)
N_examples = len(balanced['sentiments'])
n_neutral/N_examples

token_ids = [[vocab[word] for word in message] for message in balanced['messages']]
sentiments = balanced['sentiments']


import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = nn.Dropout(dropout)
        
        # Setup embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.fc = nn.Linear(lstm_size, output_size)
        self.soft = nn.LogSoftmax()


    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        #hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        
        return hidden

    def forward(self, nn_input, hidden_state):
        nn_in = nn_input.long().cuda()
        #print('NN Input: ' + str(nn_in.size()))
        embeds = self.embedding(nn_in)
        #print('embed dim: ' + str(embeds.size()))
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        lstm_out = lstm_out[-1,:,:]
        _out = self.dropout(lstm_out)
        _out = self.fc(_out)
        logps = self.soft(_out)
        #print('Output size: ' + str(logps.size()))
        
        return logps, hidden_state

model = TextClassifier(len(vocab), 10, 6, 5, dropout=0.1, lstm_layers=2)
model.embedding.weight.data.uniform_(-1, 1)
model = model.cuda()
input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
hidden = model.init_hidden(4)

logps, _ = model.forward(input, hidden)
print(logps)

def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor

split_index = int(len(token_ids)*0.8)

train_features = token_ids[:split_index]
valid_features = token_ids[split_index:]
train_labels = sentiments[:split_index]
valid_labels = sentiments[split_index:]

print('Train features count: ' + str(len(train_features)))
print('Train labels count: ' + str(len(train_labels)))



text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=32))) 
print('text_batch:', text_batch.size()) 
print('labels:', labels.size()) 
model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
hidden = model.init_hidden(text_batch.size(1))
model = model.cuda()
logps, hidden = model.forward(text_batch, hidden)



#TRAINING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(len(vocab)+1, 1024, 128, 5, lstm_layers=1, dropout=0.2)
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)

epochs = 4
batch_size = 64
learning_rate = 0.001
clip = 5
seq_len = 40

print_every = 100
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))
    
    steps = 0
    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
        steps += 1
        hidden = model.init_hidden(labels.shape[0])
        
        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
        
        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output, hidden = model.forward(text_batch, hidden)

        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # TODO Implement: Train Model
        
        if steps % print_every == 0:
            model.eval()
            val_losses = []
            accuracy = []

            with torch.no_grad():
                for text_batch, labels in dataloader(
                        train_features, train_labels, batch_size=batch_size, sequence_length=seq_len, shuffle=True):
                    text_batch, labels = text_batch.to(device), labels.to(device)
                        
                    val_h = model.init_hidden(labels.shape[0])
                    for each in val_h:
                        each.to(device)
                    output, val_h = model.forward(text_batch, val_h)
                    val_loss = criterion(output, labels)
                    val_losses.append(val_loss.item())
                    
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
                    # TODO Implement: Print metrics
            
            print("Epoch: {}/{}...".format(epoch+1, epochs),
                  "Step: {}...".format(steps),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)),
                  "Accuracy: {:.6f}".format(np.mean(accuracy)))
            model.train()



# Prediction 

def predict(text, model, vocab):
    tokens = preprocess(text)
    
    # Filter non-vocab words
    tokens = [word for word in tokens if word in vocab]
    # Convert words to ids
    tokens = [vocab[word] for word in tokens]
    token_tensor=torch.tensor(tokens)   
    # Adding a batch dimension
    text_input = token_tensor.view(-1,1)
    # Get the NN output
    
    hidden = model.init_hidden(batch_size=1)
    logps, _ = model(text_input, hidden)
    # Take the exponent of the NN output to get a range of 0 to 1 for each label.
    pred = logps.exp()
    
    return pred

text = "Google is working on self driving cars, I'm bullish on $goog"
model.eval()
model.to("cpu")
predict(text, model, vocab)

with open(os.path.join('..', '..', 'data', 'project_6_stocktwits', 'test_twits.json'), 'r') as f:
    test_data = json.load(f)


def twit_stream():
    for twit in test_data['data']:
        yield twit

next(twit_stream())



def score_twits(stream, model, vocab, universe):
    for twit in stream:

        # Get the message text
        text = twit['message_body']
        symbols = re.findall('\$[A-Z]{2,4}', text)
        score = predict(text, model, vocab)

        for symbol in symbols:
            if symbol in universe:
                yield {'symbol': symbol, 'score': score, 'timestamp': twit['timestamp']}


universe = {'$BBRY', '$AAPL', '$AMZN', '$BABA', '$YHOO', '$LQMT', '$FB', '$GOOG', '$BBBY', '$JNUG', '$SBUX', '$MU'}
score_stream = score_twits(twit_stream(), model, vocab, universe)

next(score_stream)

