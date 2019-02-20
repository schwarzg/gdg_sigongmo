#Imports for the entire Kernel
import pandas as pd
import numpy as np
import csv
import itertools
import nltk


vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

df = pd.read_csv("winemag-data-130k-v2.csv")
df = df["description"]

train_string=df.str.cat()

train_string=train_string.decode('utf-8').lower()

# Split full comment into sentences
sentences=nltk.sent_tokenize(train_string)[:1000]

# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Parsed %d sentences." % (len(sentences))	

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

#One-hot vectorized
#ind_sent=np.append([[word_to_index[w] for w in sent] for sent in tokenized_sentences],[])
X_tr=[]
for sent in tokenized_sentences:
	X_tr=X_tr+[word_to_index[w] for w in sent] 
print X_tr[:1000]
