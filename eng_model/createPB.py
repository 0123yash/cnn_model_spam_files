import os, argparse

import tensorflow as tf
#PLEASE INPUT THE MODEL DIRECTORY
def freeze_graph(model_dir, output_node_names):    
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    #     if not output_node_names:
    #         print("You need to supply the name of a node to --output_node_names.")
    #         return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    output_graph = '/data/cnn-text-classification-tf/runs/1553854145/frozen_model.pb'
    #output_graph = '/data/cnn-text-classification-tf/spamtest_yash/check_save_1546251097/frozen_model.pb'
    #output_graph = '/data/cnn-text-classification-tf/spamtest_yash/saved_checkpoints/model_1Feb_pure_hindi_1549005018_39k_iters/frozen_model.pb'
    #output_graph  = '/data1/spam_filter/cnn_saved_models/saved_checkpoints/model_8Feb_1549631544_145k_iters/frozen_model.pb'

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
            #[n.name for n in tf.get_default_graph().as_graph_def().node]
	    #variable_names_blacklist=['import/embedding/W/Assign']
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("saving the final frozen_pb at : ", output_graph)
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

#model_dir = "/opt/workarea/TextCNN/cnn-text-classification-tf/runs/1542372265/checkpoints"
model_dir = "/data/cnn-text-classification-tf/runs/1553854145/checkpoints/"
#model_dir = "/data/cnn-text-classification-tf/spamtest_yash/check_save_1546251097/checkpoints/"
#model_dir = "/data/cnn-text-classification-tf/spamtest_yash/saved_checkpoints/model_1Feb_pure_hindi_1549005018_39k_iters/checkpoints/"
#model_dir = "/data1/spam_filter/cnn_saved_models/saved_checkpoints/model_8Feb_1549631544_145k_iters/checkpoints/"

#output_node_names = ""
output_node_names = "output/predictions/dimension,output/predictions"
freeze_graph(model_dir, output_node_names)

