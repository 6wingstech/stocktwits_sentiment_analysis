# stocktwits_sentiment_analysis

Framework for training a neural network for generating bullish/bearish bias from tweets. Training data (tweets) should be entered in the form `{'message_body': 'TWEET', {'sentiment': 2}, 'timestamp': timestamp}` Sentiment should range from -2 to 2 (5 categories), with 0 being netural and 2 being bullish.

On a dataset of 1,000,000 tweets, the neural network should take about an hour to train on a GPU, and have an accuracy north of 80%


