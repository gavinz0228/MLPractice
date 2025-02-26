import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding,Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import keras
import json

def exclude_long_sentences(source_path, output_path, word_limit = 20, line_separator = '\n', convo_separator = "\n\n\n\n"):
    wfile = open(output_path, "w")
    rfile = open(source_path, "r")
    content = rfile.read()
    convos = content.split(convo_separator)
    for convo in convos:
        sentences_to_write = []
        sentences = convo.split(line_separator)
        under_limit = True
        for sen in sentences:
            words = sen.split()
            if len(words) > word_limit:
                under_limit = False
                break
            sentences_to_write.append(sen)
        if under_limit:
            sl = len(sentences_to_write)
            for i in range(sl):
                wfile.write(sentences_to_write[i])
                if i == sl - 1:
                    wfile.write(convo_separator)
                else:
                    wfile.write(line_separator)
    wfile.close()
    rfile.close()
def read_sentences(file_path,line_separator = '\n', convo_separator = "\n\n\n\n"):
    sentences = []
    rfile = open(file_path, "r")
    content = rfile.read()
    convos = content.split(convo_separator)
    for convo in convos:
        sens = convo.split(line_separator)
        for sen in sens:
            words = sen.lower().replace('?', '').replace('!', '').replace( '.', '').replace( ',', '').replace( '-', '').replace(';', '').split()
            sentences.append(words)
    return sentences
def read_lines(file_path, separator, name_idx, content_idx, start_line = 0, limit = 10000):
    ln = 1
    line_added = 0
    prev_name = None
    sentences = []
    abb_word = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'eos', 'bos']
    orig_word = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '']

    for line in open(file_path, 'r', encoding="utf-8"):

        if ln >= start_line + limit:
            if line_added % 2 == 0:
                break
            else:
                limit += 1
        elif ln < start_line:
            ln += 1
            continue
        ln += 1
        items = line.split(separator)
        name = items[name_idx].lower()
        content = items[content_idx].lower()
        for i in range(len(abb_word)):
            content = content.replace(abb_word[i], orig_word[i])
        words = content.split()
        if prev_name != name:
            sentences.append(words)
            line_added +=1
        else:
            #print(prev_name, "spoke again")
            sentences[-1].extend(words )
        prev_name = name
    #print(len(sentences))
    if len(sentences) % 2 != 0:
        sentences.pop();
    return sentences
def build_vocab_dict(sentences, padding_char, unknown_char):
    word_freq = nltk.FreqDist(np.hstack(sentences))
    word_freq.pop(padding_char, None)
    word_freq.pop(unknown_char, None)
    vocab = word_freq.most_common(VOCAB_SIZE - 2)

    vocab.insert(0, (padding_char,1))
    vocab.append( (unknown_char,1))
    
    vocab_dict = {pair[0]: id for id, pair in enumerate(vocab)}
    
    idx_dict = {idx:word for word, idx in vocab_dict.items()}
    print(idx_dict)
    return vocab_dict, idx_dict


def save_vocab(vocab_dict):
    f = open(vocab_dict, 'w') 
    f.write(json.dumps(vocab_dict)) 
    f.close() 
def load_vocab(vocab_dict):
    f = open(vocab_dict, 'r') 
    vocab_dict = json.loads(f.read() )
    f.close() 
    idx_dict = idx_dict = {idx:word for word, idx in vocab_dict.items()}
    return vocab_dict, idx_dict
def sentence_to_vec(sentences, vocab_dict, unknown_char, sentence_length):
    l = len(sentences)
    vec = []
    #unk_idx = vocab_dict[unknown_char]

    for sen in sentences:
        word_list = [vocab_dict[x] for x in sen if x in vocab_dict ][:sentence_length ]
        vec.append( word_list )

    padded = pad_sequences(vec, maxlen=sentence_length, dtype='int32')
    return padded
def to_one_hot(vec, sentence_length, vocab_length):
    print((len(vec), sentence_length, vocab_length))
    res = np.zeros((len(vec), sentence_length, vocab_length))
    for i, sen in enumerate(vec):
        for j, num in enumerate(sen):
            res[i, j, num] = 1
    return res
def vectorize_sentence(sentences, vocab_dict, vocab_size, sentence_size):

    x_sentences = [sentences[i] for i in range(len(sentences)) if i % 2 == 0]
    y_sentences = [sentences[i] for i in range(len(sentences)) if i % 2 == 1]
    
    if len(x_sentences) > len(y_sentences):
        x_sentences = x_sentences[:-1]
    elif len(y_sentences) > len(x_sentences):
        y_sentences = y_sentences[:-1]
    #print(x_sentences,x_sentences)
    x_vec = sentence_to_vec(x_sentences, vocab_dict, 'UNK', sentence_size)
    y_vec = sentence_to_vec(y_sentences, vocab_dict, 'UNK', sentence_size)

    y_vec = to_one_hot(y_vec, sentence_size, vocab_size)
    return x_vec, y_vec
def loadGloVe(filename, num_words):
    vocab = []
    file = open(filename,'r', encoding="utf8")
    i = 0
    for line in file.readlines():
        if i == num_words:
            break
        row = line.strip().split(' ')
        vocab.append( ( row[0], np.array( [ float(x) for x in row[1:]] )  ) )
        i += 1
    print('Loaded GloVe!')
    file.close()
    return vocab
def get_vocab(glove_path, vocab_size):
    prepend = [(' ', np.zeros((300)))]
    vocab = loadGloVe(glove_path, vocab_size - len(prepend))
    vocab = prepend + vocab
    vocab_to_idx = {}
    idx_to_vocab = {}
    embeding_matrix = []
    for i in range(vocab_size):
        vo = vocab[i]
        vocab_to_idx[vo[0]] = i
        idx_to_vocab[i] = vo[0]
        embeding_matrix.append(vo[1])
    embeding_matrix = np.array(embeding_matrix)
    return vocab_to_idx, idx_to_vocab, embeding_matrix