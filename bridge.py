"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by Yuhang Ming, modified form He Wang's work
"""

import os
import argparse

import sys
import time
import numpy as np
from config import Config
import model as modellib
from train import ScenesConfig

import utils
# from aligning import estimateSimilarityTransform

# # code for tensorflow v1
# import tensorflow as tf 
# from keras.backend.tensorflow_backend import set_session
# tfconfig = tf.compat.v1.ConfigProto()
# # tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# # tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
# tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.compat.v1.Session(config=tfconfig))

import tensorflow as tf
# Limit GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
print("\n##########\n", gpus)
if gpus:
  # # Restrict TensorFlow to only use the first GPU
  # try:
  #   tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  #   logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  #   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  # except RuntimeError as e:
  #   # Visible devices must be set before GPUs have been initialized
  #   print(e)

  # Currently, memory growth needs to be the same across GPUs
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

  # # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  # try:
  #   tf.config.experimental.set_virtual_device_configuration(
  #       gpus[0],
  #       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  #   logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  #   print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  # except RuntimeError as e:
  #   # Virtual devices must be set before GPUs have been initialized
  #   print(e)
else:
  print("Looks like no GPU devices found.")

print("##########\n")



from dataset import NOCSDataset


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

def load_nocs_detector(model_path, bPrint):
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Path to COCO trained weights
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

    # Load config file
    config = InferenceConfig()
    if bPrint:
        config.display()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=MODEL_DIR)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    return model

def load_dataset(data):
    config = InferenceConfig()

    # dataset directories
    camera_dir = os.path.join('data', 'camera')
    real_dir = os.path.join('data', 'real')
    vil_dir = os.path.join('data', 'vil')

    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
    # maps from coco to their own dataset
    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }

    if data == 'val':
        dataset_val = NOCSDataset(synset_names, 'val', config)
        dataset_val.load_camera_scenes(camera_dir)
        dataset_val.prepare(class_map)
        dataset = dataset_val
    elif data == 'real_test':
        dataset_real_test = NOCSDataset(synset_names, 'test', config)
        dataset_real_test.load_real_scenes(real_dir)
        dataset_real_test.prepare(class_map)
        dataset = dataset_real_test
    elif data == 'vil':
        dataset_vil = NOCSDataset(synset_names, 'vil', config)
        dataset_vil.load_vil_scenes(vil_dir)
        dataset_vil.prepare(class_map)
        dataset = dataset_vil
    else:
        assert False, "Unknown data resource."

    return dataset

def detection_x1(nocs_detector, image):
    start = time.time()
    # print(image.shape)
    # print(type(image[0, 0, 0]))
    # print(depth.shape)
    # print(type(depth[0, 0]))
    ### !!!!! Consider pass intrinsic in !!!!
    intrinsics = np.array([[580, 0, 319.5], [0, 580, 239.5], [0, 0, 1]])

    ## detection
    detect_result = nocs_detector.detect([image], verbose=0)
    r = detect_result[0]
    num_of_instance = len(r['class_ids'])

    # print('in Python')
    # print(num_of_instance)

    # assuming one object per class, remove rebundant detections
    # ind_to_remove = []
    # for i in range(num_of_instance):
    #     for j in range(num_of_instance):
    #         if i == j:
    #             continue
    #         if r['class_ids'][i] == r['class_ids'][j]:
    #             # compare the score
    #             if r['scores'][i] < r['scores'][j]:
    #                 ind_to_remove.append(i)
    #                 break
    # r['rois'] = np.delete(r['rois'], ind_to_remove, axis=0)     # Final dim: (# of detection, 4)            4 - parameters for 2D bounding box
    # r['class_ids'] = np.delete(r['class_ids'], ind_to_remove)
    # r['scores'] = np.delete(r['scores'], ind_to_remove)
    # r['masks'] = np.delete(r['masks'], ind_to_remove, axis=2)
    # r['coords'] = np.delete(r['coords'], ind_to_remove, axis=2) # Final dim: (480, 640, # of detection, 3)  3 - x, y, z coordinates of the NOCS map
    # num_of_instance = len(r['class_ids'])

    # print(r['class_ids'])
    # print(r['scores'])
    # print(type(r['scores'][0]))
    # print(type(r['scores'][0].astype(np.float32)))
    # print('end Python')

    # By default, reshape in the first dimension (row)
    # store labels, scores, and 2d bounding boxes
    pred_labels = r['class_ids'].astype(np.int32)                              # (# of detection, )
    pred_scores = r['scores'].astype(np.float32)                               # (# of detection, )
    pred_masks = r['masks'].transpose(2,0,1).reshape(-1).astype(np.int32)      # Ori: (480, 640, # of detection), T: (#, 480, 640)
    pred_coord = r['coords'].transpose(2,3,0,1).reshape(-1).astype(np.float32) # Ori: (480, 640, # of detection, 3), T: (#, 3, 480, 640)
    if num_of_instance == 0:
        print('!!!! No instance is detected.')
    
    # print(pred_coord[0:20])

    # timing
    elapsed = time.time() - start
    print('**** Detection in Python takes {:03f}.'.format(elapsed))

    return (pred_labels, pred_scores, pred_masks, pred_coord)


if __name__ == '__main__':
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='detect', type=str, help="detect/eval")
    parser.add_argument('--use_regression', dest='use_regression', action='store_true')
    parser.add_argument('--use_delta', dest='use_delta', action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
    parser.add_argument('--data', type=str, help="vil_test", default='vil')
    parser.add_argument('--gpu',  default='0', type=str)
    parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
    parser.add_argument('--num_eval', type=int, default=-1)

    parser.set_defaults(use_regression=False)
    parser.set_defaults(draw=False)
    parser.set_defaults(use_delta=False)
    args = parser.parse_args()

    mode = args.mode
    data = args.data
    ckpt_path = args.ckpt_path
    
    ## !!!! Consider, how to pass following variables into "InferenceConfig" !!!!
    use_regression = args.use_regression
    use_delta = args.use_delta
    num_eval = args.num_eval

    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    print('Using GPU {}.'.format(args.gpu))

    # load the detector
    model = load_nocs_detector(ckpt_path, True)

    # Test on a single image
    print('*'*50)
    image_start = time.time()
    image_path = "0.png"
    print(image_path)

    import cv2
    image= cv2.imread(image_path)[:, :, :3]
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_h, img_w, img_d = image.shape
    print(img_h, img_w, img_d)
    print(type(image))
    image = image[:, :, ::-1]
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # perform detection on a pair on images
    labels, scores, masks, coords = detection_x1(model, image)
    print(labels.shape)
    print(scores.shape)
    print(masks.shape)
    print(coords.shape)
    print("label - score")
    for i in range(labels.shape[0]):
        print(labels[i], "-", scores[i])

    elapsed = time.time() - image_start
    print('Takes {} to finish this image.'.format(elapsed))

    # visualise detection results
    num_of_detect = labels.shape[0]
    vi_masks = masks.reshape(num_of_detect, img_h, img_w)
    vi_coords = coords.reshape(num_of_detect, 3, img_h, img_w)
    
    fused_mask = np.zeros([img_h,img_w])
    fused_coord = np.zeros([img_h, img_w, 3])
    for i in range(num_of_detect):
        tmp_mask = vi_masks[i, :, :].astype(np.uint8) * 255
        fused_mask = fused_mask + tmp_mask
        tmp_coord = vi_coords[i, :, :, :].transpose(1, 2, 0)
        fused_coord = fused_coord + tmp_coord

    cv2.imshow("img", image)
    cv2.imshow("masks", fused_mask)
    cv2.imshow("coords", fused_coord)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
