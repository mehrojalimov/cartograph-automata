#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Object Detection with Webots Camera
This script uses a TensorFlow object detection model to process frames from a Webots robot camera.

Based on code by Rokawoo
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from controller import Robot
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# User Configuration Variables
TIME_STEP = 32
MODEL_PATH = "PATH_TO_YOUR_MODEL"  # Replace with the path to your model directory
CHECKPOINT_NUMBER = 14  # Replace with your checkpoint number
LABEL_MAP_PATH = "PATH_TO_YOUR_LABELMAP"  # Replace with the path to your label map
DETECTION_THRESHOLD = 0.7  # Minimum confidence score for detecting objects

def main():
    # Initialize Webots robot
    robot = Robot()
    
    # Get camera and enable it
    camera = robot.getDevice("camera rgb")  # Using the RGB camera from your robot
    camera.enable(TIME_STEP)
    
    # Get the width and height of the camera
    width = camera.getWidth()
    height = camera.getHeight()
    print(f"Camera resolution: {width}x{height}")
    
    # Load the object detection model
    detection_function = load_model(MODEL_PATH, CHECKPOINT_NUMBER)
    
    # Load the label map
    category_index = label_map_util.create_category_index_from_labelmap(
        LABEL_MAP_PATH, use_display_name=True)
    
    # Create window for displaying the camera feed with detections
    cv2.namedWindow("Webots Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webots Object Detection", 800, 600)
    
    print("Starting object detection. Press 'q' in the CV window to exit.")
    
    # Main control loop
    try:
        while robot.step(TIME_STEP) != -1:
            # Get image from camera
            img = camera.getImage()
            if img is None:
                print("No image received from camera")
                continue
            
            # Convert Webots image to OpenCV format
            img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
            # Perform object detection
            detections = detect_objects(detection_function, img_array)
            
            # Visualize the detections
            image_with_detections = visualize_detections(img_array, detections, category_index)
            
            # Display the image with detections
            cv2.imshow("Webots Object Detection", image_with_detections)
            
            # Check for keyboard input to exit
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("User requested exit")
                break
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("Exit complete.")


def load_model(model_path, checkpoint_number):
    """Load the object detection model from a checkpoint."""
    print(f"Loading model from {model_path}, checkpoint {checkpoint_number}...")
    
    # Load pipeline config and build a detection model
    pipeline_config = os.path.join(model_path, 'pipeline.config')
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    
    # Restore checkpoint
    ckpt_path = os.path.join(model_path, f'ckpt-{checkpoint_number}')
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(ckpt_path).expect_partial()
    
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    
    print("Model loaded successfully")
    return detect_fn


def detect_objects(detection_function, image_np):
    """Perform object detection on the input image."""
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # Run inference
    detections = detection_function(input_tensor)
    
    # Process the detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # Convert detection classes to ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections


def visualize_detections(image_np, detections, category_index):
    """Visualize the detection results on the image."""
    # Create a copy of the image for visualization
    image_np_with_detections = image_np.copy()
    
    # Visualize the boxes and labels on the image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,  # Label offset
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=DETECTION_THRESHOLD,
        agnostic_mode=False
    )
    
    return image_np_with_detections


if __name__ == "__main__":
    main()