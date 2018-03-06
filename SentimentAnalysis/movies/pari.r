
# Loading Libraries

library(readr)
library('stringr')
library('readr')
library('wordcloud')
library('tm')
library('SnowballC')
library('RWeka')
library('RSentiment')

# loading the csv which as one column text

filmName = read_csv("P:/ssc/pari1.csv")

r1 = as.character(filmName$text) # fetching the text feild

# Data cleaning and reinsing

set.seed(100) # setting the start of a random number generator
sample = sample(r1, (length(r1))) # loading all the texts into sample
corpus = Corpus(VectorSource(list(sample))) # creating a corpus vector out of the string sample
corpus = tm_map(corpus, removePunctuation) # Removing the puntuations 
corpus = tm_map(corpus, content_transformer(tolower)) # converting the tweets to lower case
corpus = tm_map(corpus, removeNumbers) # removing the numbers from tweets
corpus = tm_map(corpus, stripWhitespace) # removing the unwanted white spaces
corpus = tm_map(corpus, removeWords, stopwords('english')) # removing the non vocab words or special charectors

corpus = tm_map(corpus, stemDocument) # steming the result set above
corpus = tm_map(corpus, removeWords, c('quit','haunt','sleepless','scary','devil','evil','fear','horror','scare','dangerous')) # removing non relevant words for pari

dtm_up = DocumentTermMatrix(VCorpus(VectorSource(corpus[[1]]$content))) # creating a dataframe
freq_up <- colSums(as.matrix(dtm_up))  # making a matrix list of words with their counts


# Calculating positive or negative Sentiments

sentiments_up = calculate_sentiment(names(freq_up)) 
sentiments_up = cbind(sentiments_up, as.data.frame(freq_up))
sent_pos_up = sentiments_up[sentiments_up$sentiment == 'Positive',]
sent_neg_up = sentiments_up[sentiments_up$sentiment == 'Negative',]

cat("We have far lower negative Sentiments: ",sum(sent_neg_up$freq_up)," than positive: ",sum(sent_pos_up$freq_up))

# ploting the word cloud graph

#positive
layout(matrix(c(1, 2), nrow=2), heights=c(1, 4))
par(mar=rep(0, 4))
plot.new()
set.seed(100)
wordcloud(sent_pos_up$text,sent_pos_up$freq,min.freq=10,colors=brewer.pal(6,"Dark2"))

# Negative

plot.new()
set.seed(100)
wordcloud(sent_neg_up$text,sent_neg_up$freq, min.freq=2,colors=brewer.pal(6,"Dark2"))



