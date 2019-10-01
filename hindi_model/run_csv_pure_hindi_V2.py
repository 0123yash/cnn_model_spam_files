#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv
from tensorflow.python.platform import gfile
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

#csvFileName = 'NBTO_10_11_12_May_1551860056_prediction'
#csvFileName = 'NBTO_16-19_May'
csvFileName = 'NBTO_22_28_May'
#csvFileName = 'NBTO_14_15_May_1551860056_prediction'
#csvFileName = 'rt-polarity-hindi_uniq_off_words_new_added_shuf'

#csvSuffix = '.neg'
csvSuffix = '.csv'
csvFilePath = 'after_train/data_to_run/' + csvFileName + csvSuffix

df = pd.read_csv(csvFilePath)
#df = pd.read_csv(csvFilePath, names=['C_T'])

#df = df[:30]

x_raw1 = df['C_T'].tolist()
x_raw = [data_helpers.clean_str_pure_hindi(sent) for sent in x_raw1]

print('shape of dataframe : ' + str(df.shape))

#print(x_raw)
print('length x_raw : ',len(x_raw))

#y_test = [1, 1]
y_test = None

# Map data into vocabulary

#model_name = 'model_1551860056_6Mar_20k'
#model_name = 'model_1557741446_20k'
model_name = 'model_1557837780_20k'
#model_name = 'model_1558012024_20k'

base_dir = "saved_models/" + model_name+ "/"

vocab_path = base_dir + 'vocab'

def my_tokenizer_func(iterator):
    return (x.split(" ") for x in iterator)

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

with tf.Session() as sess:
    model_filename = base_dir + 'frozen_model.pb'

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        # output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
        # Get the placeholders from the graph by name
        input_x = tf.get_default_graph().get_operation_by_name("import/input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = tf.get_default_graph().get_operation_by_name("import/dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = tf.get_default_graph().get_operation_by_name("import/output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = tf.get_default_graph().get_operation_by_name("import/output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            print(batch_predictions_scores[1])
            print(probabilities)
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

#save the results to a CSV

prob_arr = [prob[0] for prob in all_probabilities]

df[model_name] = prob_arr
#df['prob'] = prob_arr

SPAM_THRESHOLD = 0.7
df['isSpam_' + model_name] = df[model_name]>=SPAM_THRESHOLD

out_file_name = csvFileName + '_prediction.csv'
out_path = base_dir + out_file_name

print("Saving evaluation to {0}".format(out_path))

df.to_csv(out_path, encoding='utf-8', index=False)

