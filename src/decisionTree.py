from __future__ import print_function, division
import nltk
import easygui
import os
import random
import tkMessageBox
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer,sent_tokenize
from nltk.corpus import stopwords
from nltk import DecisionTreeClassifier, classify


ignoreWordList = stopwords.words('english')
curPath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def wordExtract(sentence):
    return [WordNetLemmatizer().lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

def folderScan(folder):
    tweets = []
    files=os.listdir(curPath+folder)
    for myFile in files:
        print(str(myFile))
        f = open(curPath+folder + myFile, 'r')
        tweets.append(f.read())
        f.close()
    return tweets 
 
def tweetFeatures(text, setting):
   if setting=='bow':
        return {word: count for word, count in Counter(wordExtract(text)).items()  if not word in ignoreWordList}
 
def train(features, samples_proportion):
    trainSize = int(len(features) * samples_proportion)
    trainSet, testSet = features[:trainSize], features[trainSize:] 
    print ("Training Set Size =" + str(len(trainSet)) + ' Tweets')
    print ("Test Set size =" + str(len(testSet)) + ' Tweets')
    # train the nbClassifier
    nbClassifier = DecisionTreeClassifier.train(trainSet)
    return testSet,nbClassifier
 
def classifyTweet(testSet,nbClassifier):
    # check how the nbClassifier performs on the test set
    print ("Accuracy on the test set = " + str(classify.accuracy(nbClassifier, testSet)))
    
    
def resultDisplay(testSet,nbClassifier):
    outputFile=open("dtOutput.txt",'w+')
    outputFile.write("Original Type\tClassified type\t\tResult Type\n")
    total=len(testSet)
    tp=tn=fp=fn=0
    for tweet in testSet:
        type=str(nbClassifier.classify(tweet[0]))
        if str(tweet[1])==type:
             part1="True"
        else:
             part1="False"          
        if type=="spam":
             part2="Positive"
        else :
             part2="Negative"
        result=part1+" "+part2 
        outputFile.write(tweet[1]+"\t\t"+type+"\t\t\t"+result+"\n")
        if result=="True Positive" :
              tp+=1
        elif result=="True Negative":
              tn+=1
        elif result=="False Positive":
              fp+=1
        else:
              fn+=1
    outputFile.write("True Positive :"+str(float(tp/total)*100)+"%"+"\n"+"True Negative :"+str(float(tn/total)*100)+"%"+"\n"+"False Positive :"+str(float(fp/total)*100)+"%"+"\n"+"False Negative :"+str(float(fn/total)*100)+"%"+"\n")
    outputFile.write("Accuracy :"+str(float((tp+tn)/total)*100)+"%\n")
    outputFile.close()                                    
        

#collecting spams and hams from folders
spam = folderScan("/Spams/")
ham = folderScan("/NormalTweets/")

#gathering all datas
tweets_bag = [(tweet,'spam') for tweet in spam]
tweets_bag += [(tweet,'ham') for tweet in ham]
#shuffling all datasets
random.shuffle(tweets_bag)

#all_words = set(word.lower() for passage in tweets_bag for word in word_tokenize(passage[0]))
#for storing train features
f_out=open(curPath+"/featuresDT.txt",'w+')
all_features = [(tweetFeatures(tweet, 'bow'), label) for (tweet, label) in tweets_bag] #now we got BOW style word count for each tweet with their label
f_out.write(str(all_features))
print ("Collected " + str(len(all_features)) + " tweets!")
testSet, nbClassifier = train(all_features, 0.8)

print(nbClassifier)

classifyTweet(testSet, nbClassifier)
resultDisplay(testSet,nbClassifier)

#Testing against user data
print("Please Choose tweet file..")
tkMessageBox.showinfo("Choose Tweet","Browse to the Tweet file Please!")
path=easygui.fileopenbox()
user_file=open(path,'r+')
test=user_file.read()
#print("Now enter the test text:\n")
#test=str(raw_input())
test_features = {word: count for word, count in Counter(wordExtract(test)).items() if not word in ignoreWordList}  #work on the text_file
#to test test data :
tkMessageBox.showinfo("Output","This tweet is "+str(nbClassifier.classify(test_features)))
