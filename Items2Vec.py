import os
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
import operator
import io


class ItemsConfig(object):
    max_description = 10
    max_items_in_session = 3


def clean_text(token):
    token = token.replace(",", "")
    token = token.replace(".", "")
    token = token.replace("?", "")
    token = token.replace(":", "")
    token = token.replace(";", "")
    token = token.replace("\"", "")
    token = token.replace(")", "")
    token = token.replace("(", "")
    token = token.replace("[", "")
    token = token.replace("]", "")
    token = token.replace("}", "")
    token = token.replace("{", "")
    return token


def extract_vocabulary(dir_name, targetTextFileName="bla"):
    count = 0
    vocabulary = dict()
    for root, dirs, files in os.walk(dir_name):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            if (file.endswith("txt")):
                count = count + 1
                print("count: ", count)
                with open(dir_name + "/" + os.path.basename(root) + "/" + file) as f:
                    for line in f:
                        line = line.lower()
                        tokens = line.split()
                        for token in tokens:
                            clean_token = clean_text(token)
                            if vocabulary.has_key(clean_token):
                                vocabulary[clean_token] = vocabulary[clean_token] + 1
                            else:
                                vocabulary[clean_token] = 1

    return vocabulary


def write_vocab_2_file(filename, vocab):
    with open(filename, "w") as myfile:
        for pair in vocab:
            myfile.write(pair[0] + " " + str(pair[1]) + "\n")


def read_vocab_to_list(filename):
    return [word for line in open(filename, 'r') for word in line.split()]


def get_num_sorted(sorted_dict,token):
    count = 0
    for pair in sorted_dict:
        if pair[0] == token:
            return count
        else:
            count +=1
    return count

def get_ys(filename, vocab):
    count = 0
    ys_tokens = []
    config = ItemsConfig()
    ys = ""

    sorted_vocabulary = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

    with open(filename) as f:
        for line in f:
            line = line.lower()
            tokens = line.split()
            for token in tokens:
                token = clean_text(token)
                if token.isalpha() and count < config.max_description:
                   # print("appending token y")
                    num_token = get_num_sorted(sorted_vocabulary, token)
                    ys_tokens.append(vocab[token])
                    count += 1
    for y in ys_tokens:
        ys += str(y)+","

    return ys[:-1]


def get_xs(folder, exclude_file, vocab):
    xs = ""
    count = 0
    config = ItemsConfig()

    for root, dirs, files in os.walk(folder):
        path = root.split('/')
        print (len(path) - 1) *'---' , os.path.basename(root)
        for file in files:
            print (file, exclude_file)
            if file.endswith("txt") and (not exclude_file.endswith(file)):
                if count < config.max_items_in_session:
                    count += 1
                    xs += get_ys(folder + "/" + file, vocab) + ","

    return xs


def create_train_file(folders_list, train_filename, vocab):

    for session_folder in folders_list:
        for root, dirs, files in os.walk(session_folder):
            path = root.split('/')
            print (len(path) - 1) *'---' , os.path.basename(root)
            for file in files:
                if file.endswith("txt"):
                    print("starting ys")
                    ys = get_ys(session_folder + "/" + file, vocab)
                    print("starting xs")
                    xs = get_xs(session_folder, session_folder + "/" + file, vocab)
                    csv_line = xs + ys + "\n"
                    with open(train_filename, "a") as train_file:
                        train_file.write(csv_line)


def trainModel(train_file, vocab_size):

    train = pd.read_csv(train_file)
    Xs = train.iloc[:, :20]
    Ys = train.iloc[:, 20:]

    xs = np.array(Xs)
    ys = np.array(Ys)

    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=20))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(10, activation='sigmoid'))

    print("compile..")
    model.compile('adam', 'mse')
    print("tyrain..")
    model.fit(xs, ys)
    print("trained...")

    model.layers.pop() # Get rid of the classification layer
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    predictions = model.predict(xs) # here actually you have the item_doc-2-vec
    print(predictions)


def main():

    complete_sessions_folder = "/Users/macbook/Desktop/corpora/view_sessions"
    session_folder1 = "/Users/macbook/Desktop/corpora/view_sessions/session1"
    session_folder2 = "/Users/macbook/Desktop/corpora/view_sessions/session2"
    vocabulary_filename = "/Users/macbook/Desktop/corpora/aux_files/sessions_vocab.txt"
    train_set_filename = "/Users/macbook/Desktop/corpora/aux_files/sessions_train.txt"

    vocab = extract_vocabulary(complete_sessions_folder, vocabulary_filename)
    create_train_file([session_folder1, session_folder2], train_set_filename, vocab)

    trainModel("/Users/macbook/Desktop/corpora/aux_files/sessions_train.txt", len(vocab.keys()))



if __name__ == "__main__":
    main()