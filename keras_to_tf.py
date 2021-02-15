
'''
Input arguments:

num_output: this value has nothing to do with the number of classes, batch_size, etc., 
and it is mostly equal to 1. If the network is a **multi-stream network** 
(forked network with multiple outputs), set the value to the number of outputs.

quantize: if set to True, use the quantize feature of Tensorflow
(https://www.tensorflow.org/performance/quantization) [default: False]

use_theano: Thaeno and Tensorflow implement convolution in different ways.
When using Keras with Theano backend, the order is set to 'channels_first'.
This feature is not fully tested, and doesn't work with quantizization [default: False]

input_fld: directory holding the keras weights file [default: .]

output_fld: destination directory to save the tensorflow files [default: .]

input_model_file: name of the input weight file [default: 'model.h5']

output_model_file: name of the output weight file [default: args.input_model_file + '.pb']

graph_def: if set to True, will write the graph definition as an ascii file [default: False]

output_graphdef_file: if graph_def is set to True, the file name of the 
graph definition [default: model.ascii]

output_node_prefix: the prefix to use for output nodes. [default: output_node]

'''


# Parse input arguments

import argparse
parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_fld', action="store", 
                    dest='input_fld', type=str, default='.')
parser.add_argument('-output_fld', action="store", 
                    dest='output_fld', type=str, default='')
parser.add_argument('-input_model_file', action="store", 
                    dest='input_model_file', type=str, default='model.h5')
parser.add_argument('-output_model_file', action="store", 
                    dest='output_model_file', type=str, default='')
# parser.add_argument('-output_graphdef_file', action="store", 
#                     dest='output_graphdef_file', type=str, default='model.ascii')
# parser.add_argument('-num_outputs', action="store", 
#                     dest='num_outputs', type=int, default=1)
# parser.add_argument('-graph_def', action="store", 
#                     dest='graph_def', type=bool, default=False)
# parser.add_argument('-output_node_prefix', action="store", 
#                     dest='output_node_prefix', type=str, default='output_node')
# parser.add_argument('-quantize', action="store", 
#                     dest='quantize', type=bool, default=False)
# parser.add_argument('-theano_backend', action="store", 
#                     dest='theano_backend', type=bool, default=False)
parser.add_argument('-f')
args = parser.parse_args()
# parser.print_help()
print('input args: ', args)
print('')

# if args.theano_backend is True and args.quantize is True:
#     raise ValueError("Quantize feature does not work with theano backend.")


# load nocs detector in keras model
import os
from config import Config
import model as modellib
from train import ScenesConfig
class InferenceConfig(ScenesConfig):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    COORD_USE_REGRESSION = False
    if COORD_USE_REGRESSION:
        COORD_REGRESS_LOSS   = 'Soft_L1' 
    else:
        COORD_NUM_BINS = 32
    COORD_USE_DELTA = False

    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

# Load config file
config = InferenceConfig()
config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=config,
                          model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
model_path = args.input_fld+ '/' + args.input_model_file
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)
print("Model LOADED.")
print('')

model.keras_model.summary()
print("")

# print node names
input_node_names = [node.op.name for node in model.keras_model.inputs]
print('Input nodes names are: ', input_node_names)
pred_node_names = [node.op.name for node in model.keras_model.outputs]
print('Output nodes names are: ', pred_node_names)
print('')


# convert variables to constants and save
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

sess = K.get_session()
output_fld =  args.input_fld if args.output_fld == '' else args.output_fld
if args.output_model_file == '':
    args.output_model_file = args.input_model_file + '.pbtxt'

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    
graph_io.write_graph(constant_graph, output_fld, args.output_model_file, as_text=True)
print('saved the freezed graph (ready for inference) at: ', output_fld+'/' +args.output_model_file)


#############################################
# # initialize

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path
# from tensorflow.keras import backend as K

# output_fld =  args.input_fld if args.output_fld == '' else args.output_fld
# if args.output_model_file == '':
#     args.output_model_file = str(Path(args.input_model_file).name) + '.pb'
# Path(output_fld).mkdir(parents=True, exist_ok=True)    
# weight_file_path = str(Path(args.input_fld) / args.input_model_file)


# # Load keras model and rename output

# K.set_learning_phase(0)
# if args.theano_backend:
#     K.set_image_data_format('channels_first')
# else:
#     K.set_image_data_format('channels_last')

# try:
#     net_model = load_model(weight_file_path)
# except ValueError as err:
#     print('''Input file specified ({}) only holds the weights, and not the model defenition.
#     Save the model using mode.save(filename.h5) which will contain the network architecture
#     as well as its weights. 
#     If the model is saved using model.save_weights(filename.h5), the model architecture is 
#     expected to be saved separately in a json format and loaded prior to loading the weights.
#     Check the keras documentation for more details (https://keras.io/getting-started/faq/)'''
#           .format(weight_file_path))
#     raise err
# # num_output = args.num_outputs
# # pred = [None]*num_output
# # pred_node_names = [None]*num_output
# # for i in range(num_output):
# #     pred_node_names[i] = args.output_node_prefix+str(i)
# #     pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
# # num_output =  len(net_model.output_names)
# # pred_node_names = [None]*num_output
# # pred = [None]*num_output
# # # pred_node_names = net_model.output_names
# # for i in range(num_output):
# #     pred_node_names[i] = args.output_node_prefix+str(i)
# #     pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
# input_node_names = [node.op.name for node in net_model.inputs]
# print('Input nodes names are: ', input_node_names)
# pred_node_names = [node.op.name for node in net_model.outputs]
# print('Output nodes names are: ', pred_node_names)

# # print("net_model.input.op.name:", net_model.input.op.name)
# # print("net_model.output.op.name:", net_model.output.op.name)
# # print("net_model.input_names:", net_model.input_names)
# # print("net_model.output_names:", net_model.output_names)

# # [optional] write graph definition in ascii

# sess = K.get_session()

# if args.graph_def:
#     f = args.output_graphdef_file 
#     tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
#     print('saved the graph definition in ascii format at: ', str(Path(output_fld) / f))


# # convert variables to constants and save

# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io
# if args.quantize:
#     from tensorflow.tools.graph_transforms import TransformGraph
#     transforms = ["quantize_weights", "quantize_nodes"]
#     transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
#     constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
# else:
#     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    
# graph_io.write_graph(constant_graph, output_fld, args.output_model_file, as_text=False)
# print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / args.output_model_file))
