import nltk
import csv
import itertools
import numpy as np
from theano.tensor.nnet.nnet import softmax

vocabulary_size = 100
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

class RNN(object):
    #RNN.forward_propagation = forward_propagation
    #RNN.predict = predict

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def create_dataset(self):
        print "Reading CSV file..."
        with open('reddit-comments-2015-08.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
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

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
        print X_train[0]
        print y_train[0]
        return  X_train,y_train

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            print self.V.dot(s[t])
            o[t]=softmax(self.V.dot(s[t]))
        return [o, s]


    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))

        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        print N
        return self.calculate_total_loss(x,y)/N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]


np.random.seed(10)
model = RNN(vocabulary_size)
X_train,y_train=model.create_dataset()
o, s = model.forward_propagation(X_train[10])
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])
#predictions=model.predict(X_train)
#print predictions
