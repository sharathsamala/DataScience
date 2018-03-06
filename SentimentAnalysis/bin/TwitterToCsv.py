import tweepy
import csv
import pandas as pd

####input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

# Open/Create a file to append data
csvFile = open('pari1.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

count = 0
for tweet in tweepy.Cursor(api.search,q="#Pari",count=500,
                           lang="en",
                           since="2018-03-02").items():
    #print (tweet.created_at, tweet.text)
    count = count+1
    print (count, " " +str(tweet.created_at))
    csvWriter.writerow([tweet.text.encode('utf-8')])
        
    if count > 9999 : 
        break
