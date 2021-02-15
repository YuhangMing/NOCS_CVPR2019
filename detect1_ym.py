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

import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
tfconfig = tf.compat.v1.ConfigProto()
# tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.compat.v1.Session(config=tfconfig))


from dataset import NOCSDataset
import cv2

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

model = load_nocs_detector("logs/nocs_rcnn_res50_bin32.h5", True)

image = cv2.imread("0.png")[:, :, :3]
image = image[:, :, ::-1]

detect_result = model.detect([image], verbose=0)


# def detection_x1(nocs_detector, image, depth):
#     # print(image.shape)
#     # print(type(image[0, 0, 0]))
#     # print(depth.shape)
#     # print(type(depth[0, 0]))
#     ### !!!!! Consider pass intrinsic in !!!!
#     intrinsics = np.array([[580, 0, 319.5], [0, 580, 239.5], [0, 0, 1]])

#     ## detection
#     start = time.time()
#     detect_result = nocs_detector.detect([image], verbose=0)
#     r = detect_result[0]
#     elapsed = time.time() - start
#     # assuming one object per class, remove rebundant detections
#     num_of_instance = len(r['class_ids'])
#     ind_to_remove = []
#     # for i in range(num_of_instance):
#     #     for j in range(num_of_instance):
#     #         if i == j:
#     #             continue
#     #         if r['class_ids'][i] == r['class_ids'][j]:
#     #             # compare the score
#     #             if r['scores'][i] < r['scores'][j]:
#     #                 ind_to_remove.append(i)
#     #                 break
#     r['rois'] = np.delete(r['rois'], ind_to_remove, axis=0)     # Final dim: (# of detection, 4)            4 - parameters for 2D bounding box
#     r['class_ids'] = np.delete(r['class_ids'], ind_to_remove)
#     r['scores'] = np.delete(r['scores'], ind_to_remove)
#     r['masks'] = np.delete(r['masks'], ind_to_remove, axis=2)
#     r['coords'] = np.delete(r['coords'], ind_to_remove, axis=2) # Final dim: (480, 640, # of detection, 3)  3 - x, y, z coordinates of the NOCS map
#     num_of_instance = len(r['class_ids'])
#     # store labels, scores, and 2d bounding boxes
#     print('Detection in Python takes {:03f}.'.format(elapsed))
#     pred_labels = r['class_ids']    # Final dim: (# of detection, )
#     pred_labels = pred_labels.astype(np.int64)    
#     pred_scores = r['scores']       # Final dim: (# of detection, )
#     pred_masks = r['masks']         # Final dim: (480, 640, # of detection)     480 - height; 640 - width
#     pred_masks = pred_masks.transpose(2,0,1).reshape(-1).astype(np.int32)
#     pred_coord = r['coords'].transpose(2,3,0,1).reshape(-1) # reshape to (# of detection, 3, 480, 640)
#     if num_of_instance == 0:
#         print('!!!! No instance is detected.')

#     return (pred_labels, pred_scores, pred_masks, pred_coord)

# def detection_x1_oriShape(nocs_detector, image, depth):
#     # print(image.shape)
#     # print(type(image[0, 0, 0]))
#     # print(depth.shape)
#     # print(type(depth[0, 0]))
#     ### !!!!! Consider pass intrinsic in !!!!
#     intrinsics = np.array([[580, 0, 319.5], [0, 580, 239.5], [0, 0, 1]])

#     ## detection
#     start = time.time()
#     detect_result = nocs_detector.detect([image], verbose=0)
#     r = detect_result[0]
#     elapsed = time.time() - start
#     # assuming one object per class, remove rebundant detections
#     num_of_instance = len(r['class_ids'])
#     ind_to_remove = []
#     # for i in range(num_of_instance):
#     #     for j in range(num_of_instance):
#     #         if i == j:
#     #             continue
#     #         if r['class_ids'][i] == r['class_ids'][j]:
#     #             # compare the score
#     #             if r['scores'][i] < r['scores'][j]:
#     #                 ind_to_remove.append(i)
#     #                 break
#     r['rois'] = np.delete(r['rois'], ind_to_remove, axis=0)     # Final dim: (# of detection, 4)            4 - parameters for 2D bounding box
#     r['class_ids'] = np.delete(r['class_ids'], ind_to_remove)
#     r['scores'] = np.delete(r['scores'], ind_to_remove)
#     r['masks'] = np.delete(r['masks'], ind_to_remove, axis=2)
#     r['coords'] = np.delete(r['coords'], ind_to_remove, axis=2) # Final dim: (480, 640, # of detection, 3)  3 - x, y, z coordinates of the NOCS map
#     num_of_instance = len(r['class_ids'])
#     # store labels, scores, and 2d bounding boxes
#     print('Detection in Python takes {:03f}.'.format(elapsed))
#     pred_labels = r['class_ids']    # Final dim: (# of detection, )
#     pred_labels = pred_labels   
#     pred_scores = r['scores']       # Final dim: (# of detection, )
#     pred_masks = r['masks']         # Final dim: (480, 640, # of detection)     480 - height; 640 - width
#     pred_coord = r['coords']
#     if num_of_instance == 0:
#         print('!!!! No instance is detected.')

#     return (pred_labels, pred_scores, pred_masks, pred_coord)


# if __name__ == '__main__':
#     # parse the command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', default='detect', type=str, help="detect/eval")
#     parser.add_argument('--use_regression', dest='use_regression', action='store_true')
#     parser.add_argument('--use_delta', dest='use_delta', action='store_true')
#     parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
#     parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
#     parser.add_argument('--gpu',  default='0', type=str)
#     parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")
#     parser.add_argument('--num_eval', type=int, default=-1)
#     parser.add_argument('--img_path', type=str, default='data/')
#     parser.add_argument('--img_id', type=int, default=0)

#     parser.set_defaults(use_regression=False)
#     parser.set_defaults(draw=False)
#     parser.set_defaults(use_delta=False)
#     args = parser.parse_args()

#     mode = args.mode
#     data = args.data
#     ckpt_path = args.ckpt_path
#     img_path = args.img_path
#     img_id = args.img_id
    
#     ## !!!! Consider, how to pass following variables into "InferenceConfig" !!!!
#     use_regression = args.use_regression
#     use_delta = args.use_delta
#     num_eval = args.num_eval

#     os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
#     print('Using GPU {}.'.format(args.gpu))

#     # load the detector
#     model = load_nocs_detector(ckpt_path, True)

#     # # load the test images
#     # dataset = load_dataset(data)

   
#     print('*'*50)
#     image_start = time.time()

#     # load the image pair
#     image_path = img_path + str(img_id) + '_color.png'
#     image = cv2.imread(image_path)[:, :, :3]
#     image = image[:, :, ::-1]
#     # If grayscale. Convert to RGB for consistency.
#     if image.ndim != 3:
#         image =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     depth_path = img_path + str(img_id) + '_depth.png'
#     depth = cv2.imread(depth_path, -1)
#     if len(depth.shape) == 3:
#         # This is enco ded depth image, let's convert
#         depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
#         depth16 = depth16.astype(np.uint16)
#     elif len(depth.shape) == 2 and depth.dtype == 'uint16':
#         depth16 = depth
#     else:
#         assert False, '[ Error ]: Unsupported depth type.'

#     # perform detection on a pair on images
#     labels, scores, masks, coord = detection_x1_oriShape(model, image, depth)

#     print(labels.shape)
#     print(labels)
#     print(scores.shape)
#     print(scores)
#     print(masks.shape)
#     print(coord.shape)

#     elapsed = time.time() - image_start
#     print('Takes {} to finish this image.'.format(elapsed))
#     print('\n')

#     # draw the detected mask
#     draw_image = image.copy()
#     num_instances = len(labels)
#     palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
#     for i in range(num_instances):
#         # color
#         color = labels[i] * palette
#         color = (color % 255)
#         color = tuple([int(x) for x in color])

#         # contour
#         mask = masks[:, :, i]
#         #mask = mask[:, :, np.newaxis]
#         #mask = np.repeat(mask, 3, axis=-1)
#         contours, hierarchy = cv2.findContours(
#             mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#         )

#         #draw
#         draw_image = cv2.drawContours(draw_image, contours, -1, color, 3)
#         cv2.imwrite(str(img_id)+'_0.5_output.png', draw_image[:, :, ::-1])