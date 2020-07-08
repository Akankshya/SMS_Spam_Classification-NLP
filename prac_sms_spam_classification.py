from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import numpy as np
from keras.preprocessing import text, sequence
from sklearn import model_selection
from keras import layers, models, optimizers
from sklearn import metrics

spam_data = 'D:\\NLP\\AnalyticsVidhya-NLP\Handouts_v4\\Project - SMS Spam Classification\\spamdata.csv'
df = pd.read_csv(spam_data, encoding='utf-8')

lem = WordNetLemmatizer()

def pos_check(txt,family):
    tags = nltk.pos_tag(nltk.word_tokenize(txt))
    count = 0
    for tag in tags:
        tag = tag[1]
        if tag in pos_dic[family]:
            count+=1
    return count
def clean_text(text):
    cleaned = text.lower()
    punctuations = string.punctuation
    cleaned = "".join(c for c in cleaned if c not in punctuations)
    words = cleaned.split()
    stopwords_lists = stopwords.words("english")
    cleaned = [word for word in words if word not in stopwords_lists]
    cleaned = [lem.lemmatize(word, "v") for word in cleaned]
    cleaned = [lem.lemmatize(word, "n") for word in cleaned]
    cleaned = " ".join(cleaned)
    return cleaned
def train_model(classifier, feature_vector_train, label,
                feature_vector_valid, is_neural_net = False):
    
    classifier.fit(feature_vector_train, label,epochs = 100)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)
def create_cnn():
    input_layer = layers.Input((70,))
        # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return model

df["cleaned"] = df["text"].apply(lambda x: clean_text(x))

#feature engineering
df["wc"] = df["text"].apply(lambda x: len(x.split()))
df["wc_cleaned"] = df["cleaned"].apply(lambda x: len(x.split()))

df["cc"] = df["text"].apply(lambda x: len(x))
df["cc_without_spaces"] = df["text"].apply(lambda x: len(x.replace(" ","")))

df["num_dig"] = df["text"].apply(lambda x: sum([1 if w.isdigit() else 0 for w in x.split()]))

pos_dic ={"noun": ["NNP", "NN","NNS","NNPS"],
          "verb":["VBZ", "VB","VBD","VBG","VBN"]}
          
df["noun_count"] = df["text"].apply(lambda x : pos_check(x, "noun"))
df["verb_count"] = df["text"].apply(lambda x : pos_check(x, "verb"))

cvz = CountVectorizer()
cvz.fit(df["cleaned"].values)
count_vectors = cvz.transform(df["cleaned"].values)

word_tfidf = TfidfVectorizer(max_features=500)
word_tfidf.fit(df["cleaned"].values)
word_vectors_tfidf = word_tfidf.transform(df["cleaned"].values)

meta_features = ['wc','wc_cleaned','cc','cc_without_spaces',
                  'num_dig','noun_count','verb_count']
feature_set1 = df[meta_features]
train = hstack([word_vectors_tfidf,csr_matrix(feature_set1)],"csr")

target = df["label"].values
target = LabelEncoder().fit_transform(target)

print("target ",target)

trainx, valx, trainy, valy = train_test_split(train, target)

#NaiveBayes
model = naive_bayes.MultinomialNB()
model.fit(trainx, trainy)
preds = model.predict(valx)
print("Naive Bayes ",accuracy_score(preds, valy))

#LogisticRegression
model = LogisticRegression()
model.fit(trainx, trainy)
preds = model.predict(valx)
print("LogisticRegression ",accuracy_score(preds, valy))

#SVM
model = svm.SVC()
model.fit(trainx, trainy)
preds = model.predict(valx)
print("SVM ",accuracy_score(preds, valy))

#Ensemble
model = ensemble.ExtraTreesClassifier()
model.fit(trainx, trainy)
preds = model.predict(valx)
print("Ensemble ",accuracy_score(preds, valy))

embeddings_index = {}
for i,line in enumerate (open('D:\\NLP\AnalyticsVidhya-NLP\\Handouts_v4\\Project - SMS Spam Classification\\crawl-300d-2M.vec\\crawl-300d-2M.vec', encoding = "utf8")):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

token = text.Tokenizer()
token.fit_on_texts(df["text"])
word_index = token.word_index

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df["text"], target)
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen = 70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen = 70)

# create token-embedding mapping
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings",  accuracy)