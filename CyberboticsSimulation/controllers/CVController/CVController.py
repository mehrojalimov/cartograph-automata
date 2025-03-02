#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Refined Real-time Object Detection and Path Planning Controller
"""

import sys
import os
# Make sure this is the first thing in your script
sys.path.insert(0, "C:/Users/rokaw/GitProjects/Codefest2025 - TEMP/CVObjectDetection/Tensorflow/models/research")

import cv2
import numpy as np
import tensorflow as tf
from controller import Robot
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# User Configuration Variables
TIME_STEP = 32
MODEL_PATH = "C:/Users/rokaw/GitProjects/Codefest2025 - TEMP/CVObjectDetection/Tensorflow/workspace/models/codefest_2025_ssd_mobilenet_v1_fpn"
CHECKPOINT_NUMBER = 14
LABEL_MAP_PATH = "C:/Users/rokaw/GitProjects/Codefest2025 - TEMP/CVObjectDetection/Tensorflow/workspace/annotations/label_map.pbtxt"
DETECTION_THRESHOLD = 0.7
CLASSES_TO_AVOID = ['person', 'shelf']  # Classes to treat as obstacles
PATH_STEP_SIZE = 10  # Step size for path finding algorithm

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
    
    return detection_model

@tf.function
def detect_fn(detection_model, input_tensor):
    """Detection function that processes the input image and returns detections."""
    image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def extract_obstacles_from_detections(detections, category_index, height, width):
    """Extract obstacle coordinates from detection results with class names"""
    obstacles = []
    class_names = []
    line_boxes = []
    
    # Process detection data
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # Get detection data
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    
    for i in range(len(scores)):
        if scores[i] >= DETECTION_THRESHOLD:
            class_id = int(classes[i]) + 1  # Adjust for label offset
            if class_id in category_index:
                class_name = category_index[class_id]['name']
                
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                xmin_px, ymin_px = int(xmin * width), int(ymin * height)
                xmax_px, ymax_px = int(xmax * width), int(ymax * height)
                
                # Create rectangle vertices
                bbox = np.array([
                    [xmin_px, ymin_px], 
                    [xmax_px, ymin_px], 
                    [xmax_px, ymax_px], 
                    [xmin_px, ymax_px]
                ], dtype=int)
                
                if class_name in CLASSES_TO_AVOID:
                    obstacles.append(bbox)
                    class_names.append(class_name)
                elif class_name == 'line':
                    line_boxes.append(bbox)
    
    return obstacles, class_names, line_boxes, detections

def is_line_blocked(x, obstacles):
    """Check if a vertical line at x intersects any obstacle."""
    for bbox in obstacles:
        x_min, _ = bbox.min(axis=0)
        x_max, _ = bbox.max(axis=0)
        if x_min <= x <= x_max:
            return True
    return False

def find_blocked_path_segments(width, obstacles, step=PATH_STEP_SIZE):
    """Find start and end points of blocked path segments for efficient drawing
    Returns a list of tuples (start_x, end_x) for each blocked segment"""
    blocked_segments = []
    in_blocked_area = False
    start_x = None
    
    for x in range(0, width, step):
        if is_line_blocked(x, obstacles):
            if not in_blocked_area:
                start_x = x
                in_blocked_area = True
        else:
            if in_blocked_area:
                blocked_segments.append((start_x, x - step))
                in_blocked_area = False
    
    # If we end in a blocked area, close the segment
    if in_blocked_area:
        blocked_segments.append((start_x, width - step))
    
    return blocked_segments

def get_clear_paths(width, obstacles, step=PATH_STEP_SIZE):
    """Identify leftmost and rightmost x-coordinates of clear paths, centered at 0."""
    half_width = width // 2
    clear_x = []
    in_clear_area = False
    left_boundary = None

    for x in range(0, width, step):
        is_blocked = is_line_blocked(x, obstacles)
        if not is_blocked:  # Clear path
            if not in_clear_area:
                left_boundary = x - half_width  # Convert to centered coordinates
                in_clear_area = True
        else:  # Blocked path
            if in_clear_area:
                clear_x.append((left_boundary, (x - step) - half_width))  # Convert to centered coordinates
                in_clear_area = False

    if in_clear_area:  # If last section was clear, close it
        clear_x.append((left_boundary, (width - 1) - half_width))

    return clear_x

def get_central_clearance(height, width, obstacles):
    """Find vertical clearance from the bottom to the first obstacle in the center."""
    center_x = width // 2
    min_y_max = height  # Default to full height (no obstacle case)

    for bbox in obstacles:
        x_min, _, x_max, y_max = bbox.min(axis=0)[0], bbox.min(axis=0)[1], bbox.max(axis=0)[0], bbox.max(axis=0)[1]

        if x_min <= center_x <= x_max:  # If the obstacle covers the center x
            min_y_max = min(min_y_max, y_max)  # Find the closest obstacle from the bottom

    # Return the distance from the bottom to the obstacle
    return height - min_y_max if min_y_max != height else height

def draw_path_lines(image, blocked_segments, height, alpha=0.5):
    """Draw transparent lines with solid borders for blocked paths"""
    # Create a separate image for the lines
    line_img = np.zeros_like(image)
    border_img = np.zeros_like(image)
    
    for start_x, end_x in blocked_segments:
        # Calculate the range of x values for this segment
        segment_width = end_x - start_x + PATH_STEP_SIZE
        
        # Draw transparent lines for the entire blocked segment
        for x in range(start_x, end_x + PATH_STEP_SIZE, PATH_STEP_SIZE):
            cv2.line(line_img, (x, 0), (x, height), (0, 0, 255), 1)
        
        # Draw solid borders at the start and end of each blocked segment
        cv2.line(border_img, (start_x, 0), (start_x, height), (0, 0, 255), 2)
        cv2.line(border_img, (end_x + PATH_STEP_SIZE, 0), (end_x + PATH_STEP_SIZE, height), (0, 0, 255), 2)
    
    # Blend the transparent lines
    result = cv2.addWeighted(image, 1.0, line_img, alpha, 0)
    # Add the solid borders (no transparency)
    result = cv2.add(result, border_img)
    
    return result

def draw_obstacle_labels(image, obstacles, class_names):
    """Draw red bounding boxes for obstacles with red text labels at bottom"""
    for i, bbox in enumerate(obstacles):
        # Get corner points
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        
        # Draw red rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Add class label with proper name at the bottom
        class_name = class_names[i] if i < len(class_names) else "Unknown"
        # Capitalize the first letter for better appearance
        class_name = class_name.capitalize()
        
        # Calculate text size to better position the label
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            class_name, font, font_scale, font_thickness)
        
        # Position the text at the bottom of the box, centered horizontally
        text_x = x_min + (x_max - x_min - text_width) // 2
        text_y = y_max + text_height + 5  # 5 pixels below the box
        
        # Draw the text in red with no background
        cv2.putText(image, class_name, (text_x, text_y), 
                   font, font_scale, (0, 0, 255), font_thickness)  # Red text
    
    return image

def draw_line_boxes(image, line_boxes):
    """Draw yellow bounding boxes around lines with yellow text at top"""
    for bbox in line_boxes:
        # Get corner points
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        
        # Draw yellow rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow box
        
        # Add "Line" label at the top
        label = "Line"
        
        # Calculate text size for positioning
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness)
        
        # Position the text at the top of the box, centered horizontally
        text_x = x_min + (x_max - x_min - text_width) // 2
        text_y = y_min - 5  # 5 pixels above the box
        
        # Draw the text in yellow with no background
        cv2.putText(image, label, (text_x, text_y), 
                   font, font_scale, (0, 255, 255), font_thickness)  # Yellow text
    
    return image

def visualize_combined(image, obstacles, class_names, line_boxes, clear_paths, center_clearance, show_paths=True, show_boxes=True):
    """Visualize both detection and path planning results with toggleable elements"""
    vis_img = image.copy()
    height, width = image.shape[:2]
    
    # First add path planning visualization if enabled
    if show_paths:
        # Find blocked path segments for more efficient drawing
        blocked_segments = find_blocked_path_segments(width, obstacles)
        
        # Draw transparent lines with solid borders for blocked paths
        vis_img = draw_path_lines(vis_img, blocked_segments, height)
        
        # Draw center clearance indicator
        center_x = width // 2
        if center_clearance < height:  # Only draw if there's an obstacle
            cv2.line(vis_img, (center_x, height), (center_x, height - center_clearance), (0, 255, 255), 2)
            label_pos = (center_x + 5, height - center_clearance//2)
            cv2.putText(vis_img, f"{center_clearance}px", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw yellow boxes around lines and red boxes around obstacles if enabled
    if show_boxes:
        vis_img = draw_line_boxes(vis_img, line_boxes)
        vis_img = draw_obstacle_labels(vis_img, obstacles, class_names)
    
    return vis_img

def main():
    # Initialize Webots robot
    robot = Robot()
    
    # Get camera and enable it
    camera = robot.getDevice("camera rgb")
    camera.enable(TIME_STEP)
    
    # Get the width and height of the camera
    width = camera.getWidth()
    height = camera.getHeight()
    print(f"Camera resolution: {width}x{height}")
    
    # Load the object detection model
    detection_model = load_model(MODEL_PATH, CHECKPOINT_NUMBER)
    
    # Load the label map
    category_index = label_map_util.create_category_index_from_labelmap(
        LABEL_MAP_PATH, use_display_name=True)
    
    # Create window for displaying the combined view
    cv2.namedWindow("Object Detection & Path Planning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection & Path Planning", 800, 600)
    cv2.setWindowProperty("Object Detection & Path Planning", cv2.WND_PROP_TOPMOST, 1)
    
    print("Starting object detection and path planning.")
    print("Controls:")
    print("  Press 'e' to toggle path visualization")
    print("  Press 'r' to toggle bounding box visualization")
    print("  Press 'q' to exit")
    
    # Visualization toggle flags
    show_paths = True
    show_boxes = True
    
    # Frame counter for rate limiting outputs
    frame_counter = 0
    output_rate = 5  # Print path data every 5 frames
    
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
            input_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)
            detections = detect_fn(detection_model, input_tensor)
            
            # Extract obstacles and line boxes from detections
            obstacles, class_names, line_boxes, processed_detections = extract_obstacles_from_detections(
                detections, category_index, height, width)
            
            # Calculate path planning
            clear_paths = get_clear_paths(width, obstacles)
            center_clearance = get_central_clearance(height, width, obstacles)
            
            # Create combined visualization with current toggle states
            combined_img = visualize_combined(
                img_array, obstacles, class_names, line_boxes, clear_paths, center_clearance, 
                show_paths, show_boxes
            )
            
            # Print out the available paths (limiting frequency to avoid console spam)
            frame_counter += 1
            if frame_counter % output_rate == 0:
                print("\nAvailable clear paths (left, right tuples in centered coordinates):")
                for i, path in enumerate(clear_paths):
                    left, right = path
                    width_px = right - left
                    print(f"  Path {i+1}: ({left}, {right}) - width: {width_px}px")
                print(f"Center vertical clearance: {center_clearance}px")
                frame_counter = 0
            
            # Display the combined image
            cv2.imshow("Object Detection & Path Planning", combined_img)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("User requested exit")
                break
            elif key == ord('e'):
                show_paths = not show_paths
                print(f"Path visualization: {'ON' if show_paths else 'OFF'}")
            elif key == ord('r'):
                show_boxes = not show_boxes
                print(f"Bounding box visualization: {'ON' if show_boxes else 'OFF'}")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("Exit complete.")

if __name__ == "__main__":
    main()