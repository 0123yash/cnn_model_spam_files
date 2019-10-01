#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from tensorflow.python.platform import gfile
import helper_funcs
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

#csvFileName = 'only_eng_NBT_7_11_april.csv'
csvFileName = 'TOI_1_June.csv'
csvFilePath = 'spamtest_yash/data_to_run/' + csvFileName

#field_arr = ['comment_text']
#df = pd.read_csv(csvFilePath, names = field_arr) 
df = pd.read_csv(csvFilePath)
x_raw1 = df['C_T'].tolist()

#x_raw1 = helper_funcs.getListFromCsvV2(csvFilePath)
#x_raw = [data_helpers.clean_str(sent) for sent in x_raw1]
x_raw = [data_helpers.clean_str_1544432917(sent) for sent in x_raw1]

print('length x_raw : ',len(x_raw))

#y_test = [1, 1]
y_test = None

# Map data into vocabulary

base_dir = "spamtest_yash/model_12Dec_1544432917/"
#base_dir = "spamtest_yash/model_8Feb_1549626601/"

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
        # print (output_node_names)
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


print (all_predictions)

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))

#print(["{}".format(prob[0]) for prob in all_probabilities])
predictions_human_readable = np.column_stack((np.array(x_raw1),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities],
					      ["{}".format(prob[0]) for prob in all_probabilities]))


out_file_name = csvFileName + '_prediction.csv'
out_path = base_dir + out_file_name

print("Saving evaluation to {0}".format(out_path))
#print(predictions_human_readable)

with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)

