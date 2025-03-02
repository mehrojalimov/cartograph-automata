#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Threaded Real-time Object Detection and Rectangle Movement Controller
- Immediately stops when obstacles are detected in the line's path
- Seamlessly resumes movement from where it left off
"""
import sys
import os
import time
import threading
import queue
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
from MovementFunctions import init_robot, turn_right, move_forward, stop, move_backward

# User Configuration Variables
TIME_STEP = 32
MODEL_PATH = "C:/Users/rokaw/GitProjects/Codefest2025 - TEMP/CVObjectDetection/Tensorflow/workspace/models/codefest_2025_ssd_mobilenet_v1_fpn"
CHECKPOINT_NUMBER = 14
LABEL_MAP_PATH = "C:/Users/rokaw/GitProjects/Codefest2025 - TEMP/CVObjectDetection/Tensorflow/workspace/annotations/label_map.pbtxt"
DETECTION_THRESHOLD = 0.7
CLASSES_TO_AVOID = ['person', 'shelf']  # Classes to treat as obstacles
PATH_STEP_SIZE = 10  # Step size for path finding algorithm

# Movement configuration
LONG_DISTANCE = 32
SHORT_DISTANCE = 23.5
TURN_ANGLE = 90
MOVEMENT_SPEED = 1.5 * 3  # Default speed * 3 as in MovementFunctions

# Enhanced safety configuration - more sensitive to obstacles
OBSTACLE_CLEARANCE_THRESHOLD = 100  # Increased from 50 to be more sensitive
PATH_WIDTH_THRESHOLD = 150  # Increased from 100 to be more sensitive

# Global variables for thread communication
path_is_safe = True
emergency_stop = False  # New flag for immediate stopping
detection_running = True
movement_running = True
current_image = None
current_line_height = 0  # Height of detected line bounding box
safety_lock = threading.Lock()
image_lock = threading.Lock()
movement_state_lock = threading.Lock()
movement_command_queue = queue.Queue()  # Queue for movement commands
robot = None  # Global robot instance

# Movement state tracking
current_movement_index = 0
movement_in_progress = False
current_action = None
current_value = None
movement_progress = 0.0  # Progress of current movement (0.0 to 1.0)

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
    global current_line_height
    
    obstacles = []
    class_names = []
    line_boxes = []
    
    # Process detection data
    num_detections = int(detections.pop('num_detections'))
    # Pre-allocate memory for detections
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    # Get detection data - extract once to avoid repeated dictionary lookups
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    
    # Pre-filter based on threshold to avoid unnecessary processing
    valid_indices = np.where(scores >= DETECTION_THRESHOLD)[0]
    
    # Track line height
    max_line_height = 0
    
    for i in valid_indices:
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
            ], dtype=np.int32)  # Explicitly use int32 for OpenCV compatibility
            
            if class_name in CLASSES_TO_AVOID:
                obstacles.append(bbox)
                class_names.append(class_name)
            elif class_name == 'line':
                line_boxes.append(bbox)
                line_height = ymax_px - ymin_px
                max_line_height = max(max_line_height, line_height)
    
    # Update global line height
    current_line_height = max_line_height
    
    return obstacles, class_names, line_boxes, detections

def is_line_blocked(x, obstacles):
    """Check if a vertical line at x intersects any obstacle."""
    for bbox in obstacles:
        x_min, _ = bbox.min(axis=0)
        x_max, _ = bbox.max(axis=0)
        if x_min <= x <= x_max:
            return True
    return False

def is_obstacle_in_line_area(obstacles, line_boxes):
    """Check if any obstacle is directly above a line bounding box."""
    if not obstacles or not line_boxes:
        return False
    
    # For each line box, check if any obstacle is directly above it
    for line_bbox in line_boxes:
        line_x_min, line_y_min = line_bbox.min(axis=0)
        line_x_max, line_y_max = line_bbox.max(axis=0)
        
        # Define the region directly above the line
        above_x_min = line_x_min
        above_x_max = line_x_max
        above_y_min = 0  # Top of the image
        above_y_max = line_y_min  # Just above the line's top edge
        
        for obs_bbox in obstacles:
            obs_x_min, obs_y_min = obs_bbox.min(axis=0)
            obs_x_max, obs_y_max = obs_bbox.max(axis=0)
            
            # Check if obstacle horizontally overlaps with the line
            horizontal_overlap = not (obs_x_max < above_x_min or obs_x_min > above_x_max)
            
            # Check if obstacle is above the line (any part)
            is_above_line = obs_y_max < line_y_min
            
            # If both conditions are true, obstacle is in the path
            if horizontal_overlap and is_above_line:
                return True
    
    return False
    

def find_blocked_path_segments(width, obstacles, step=PATH_STEP_SIZE):
    """Find start and end points of blocked path segments for efficient drawing
    Returns a list of tuples (start_x, end_x) for each blocked segment"""
    if not obstacles:  # Early return if no obstacles
        return []
        
    blocked_segments = []
    in_blocked_area = False
    start_x = None
    
    for x in range(0, width, step):
        is_blocked = is_line_blocked(x, obstacles)
        if is_blocked:
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
    if not obstacles:  # Early return if no obstacles - full width is clear
        half_width = width // 2
        return [(-half_width, width - 1 - half_width)]
        
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
    if not obstacles:  # Early return if no obstacles
        return height
        
    center_x = width // 2
    min_y_max = height  # Default to full height (no obstacle case)
    
    for bbox in obstacles:
        x_min, _ = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        if x_min <= center_x <= x_max:  # If the obstacle covers the center x
            min_y_max = min(min_y_max, y_max)  # Find the closest obstacle from the bottom
    
    # Return the distance from the bottom to the obstacle
    return height - min_y_max if min_y_max != height else height

def check_path_safety(obstacles, line_boxes, clear_paths, center_clearance, height, width):
    """
    Determine if the current path is safe for the robot to proceed.
    
    Args:
        obstacles: List of obstacle bounding boxes
        line_boxes: Detected line bounding boxes
        clear_paths: List of clear path coordinates
        center_clearance: Vertical clearance in center of image
        height: Image height
        width: Image width
        
    Returns:
        Boolean indicating if path is safe
    """
    # First, check if any obstacle is directly above a line
    if is_obstacle_in_line_area(obstacles, line_boxes):
        return False
    
    # Check if there are no clear paths wide enough
    if not clear_paths:
        return False
    
    # Check if the center of the screen is in any clear path
    center_x = width // 2
    center_is_clear = False
    for left, right in clear_paths:
        # Convert from centered coordinates to absolute
        abs_left = center_x + left
        abs_right = center_x + right
        if abs_left <= center_x <= abs_right:
            center_is_clear = True
            break
    
    if not center_is_clear:
        # Center path is blocked
        return False
    
    # Get the widest clear path
    widest_path = max(clear_paths, key=lambda p: p[1] - p[0])
    path_width = widest_path[1] - widest_path[0]
    
    # Check if the center clearance is below threshold
    if center_clearance < OBSTACLE_CLEARANCE_THRESHOLD:
        return False
    
    # Check if path is too narrow
    if path_width < PATH_WIDTH_THRESHOLD:
        return False
    
    return True

def draw_path_lines(image, blocked_segments, height, alpha=0.5):
    """Draw transparent lines with solid borders for blocked paths"""
    if not blocked_segments:  # Skip if no blocked segments
        return image
        
    # Create a separate image for the lines (only allocate memory if needed)
    line_img = np.zeros_like(image)
    border_img = np.zeros_like(image)
    
    color_red = (0, 0, 255)  # Pre-define color tuple to avoid recreating it in loops
    
    for start_x, end_x in blocked_segments:
        # Calculate the range of x values for this segment
        # Draw transparent lines for the entire blocked segment
        for x in range(start_x, end_x + PATH_STEP_SIZE, PATH_STEP_SIZE):
            cv2.line(line_img, (x, 0), (x, height), color_red, 1)
        
        # Draw solid borders at the start and end of each blocked segment
        cv2.line(border_img, (start_x, 0), (start_x, height), color_red, 2)
        cv2.line(border_img, (end_x + PATH_STEP_SIZE, 0), (end_x + PATH_STEP_SIZE, height), color_red, 2)
    
    # Blend the transparent lines
    result = cv2.addWeighted(image, 1.0, line_img, alpha, 0)
    # Add the solid borders (no transparency)
    result = cv2.add(result, border_img)
    
    return result

def draw_obstacle_labels(image, obstacles, class_names):
    """Draw red bounding boxes for obstacles with red text labels at bottom"""
    if not obstacles:  # Skip if no obstacles
        return image
        
    color_red = (0, 0, 255)  # Pre-define color tuple
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for i, bbox in enumerate(obstacles):
        # Get corner points
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        
        # Draw red rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_red, 2)
        
        # Add class label with proper name at the bottom
        class_name = class_names[i] if i < len(class_names) else "Unknown"
        # Capitalize the first letter for better appearance
        class_name = class_name.capitalize()
        
        # Calculate text size to better position the label
        (text_width, text_height), _ = cv2.getTextSize(
            class_name, font, font_scale, font_thickness)
        
        # Position the text at the bottom of the box, centered horizontally
        text_x = x_min + (x_max - x_min - text_width) // 2
        text_y = y_max + text_height + 5  # 5 pixels below the box
        
        # Draw the text in red with no background
        cv2.putText(image, class_name, (text_x, text_y), 
                   font, font_scale, color_red, font_thickness)
    
    return image

def draw_line_boxes(image, line_boxes):
    """Draw yellow bounding boxes around lines with yellow text at top"""
    if not line_boxes:  # Skip if no line boxes
        return image
        
    color_yellow = (0, 255, 255)  # Pre-define color tuple
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label = "Line"  # Pre-define label
    
    # Pre-calculate label text size since it's the same for all boxes
    (text_width, text_height), _ = cv2.getTextSize(
        label, font, font_scale, font_thickness)
    
    for bbox in line_boxes:
        # Get corner points
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        
        # Draw yellow rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_yellow, 2)
        
        # Position the text at the top of the box, centered horizontally
        text_x = x_min + (x_max - x_min - text_width) // 2
        text_y = y_min - 5  # 5 pixels above the box
        
        # Draw the text in yellow with no background
        cv2.putText(image, label, (text_x, text_y), 
                   font, font_scale, color_yellow, font_thickness)
    
    return image

def visualize_combined(image, obstacles, class_names, line_boxes, clear_paths, center_clearance, 
                      is_safe, show_paths=True, show_boxes=True, line_height=0):
    """Visualize both detection and path planning results with toggleable elements"""
    # Only copy the image if we're going to modify it
    if (show_paths and (obstacles or center_clearance < image.shape[0])) or (show_boxes and (obstacles or line_boxes)):
        vis_img = image.copy()
    else:
        return image  # Return original if no visualization needed
    
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
            clearance_color = (0, 255, 0) if is_safe else (0, 0, 255)  # Green if safe, red if unsafe
            cv2.line(vis_img, (center_x, height), (center_x, height - center_clearance), clearance_color, 2)
            label_pos = (center_x + 5, height - center_clearance//2)
            cv2.putText(vis_img, f"{center_clearance}px", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, clearance_color, 1)
    
    # Draw yellow boxes around lines and red boxes around obstacles if enabled
    if show_boxes:
        vis_img = draw_line_boxes(vis_img, line_boxes)
        vis_img = draw_obstacle_labels(vis_img, obstacles, class_names)
    
    # Add safety status indicator
    status_text = "PATH SAFE" if is_safe else "PATH BLOCKED"
    status_color = (0, 255, 0) if is_safe else (0, 0, 255)  # Green if safe, red if unsafe
    cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Add line height information
    if line_height > 0:
        cv2.putText(vis_img, f"Line height: {line_height}px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Add movement status
    with movement_state_lock:
        if movement_in_progress:
            progress_text = f"Moving: {current_action} {current_value} - {int(movement_progress*100)}%"
        else:
            progress_text = "Movement paused"
    
    cv2.putText(vis_img, progress_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_img

def detection_thread(detection_model, category_index, camera, width, height):
    """Thread for continuous object detection and visualization"""
    global path_is_safe, emergency_stop, current_image, detection_running
    
    # Visualization toggle flags
    show_paths = True
    show_boxes = True
    
    # Create window for displaying the combined view
    cv2.namedWindow("Object Detection & Path Planning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection & Path Planning", 800, 600)
    cv2.setWindowProperty("Object Detection & Path Planning", cv2.WND_PROP_TOPMOST, 1)
    
    print("Starting detection thread...")
    
    prev_safe_state = True
    
    while detection_running and robot.step(TIME_STEP) != -1:
        # Get image from camera
        img = camera.getImage()
        if img is None:
            print("No image received from camera")
            time.sleep(0.01)  # Small delay to prevent CPU hogging
            continue
        
        # Convert Webots image to OpenCV format
        img_array = np.frombuffer(img, np.uint8).reshape((height, width, 4))
        img_rgb = img_array[:, :, :3]
        
        # Update current image for other threads
        with image_lock:
            current_image = img_rgb.copy()
        
        # Perform object detection
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_rgb, 0), dtype=tf.float32)
        detections = detect_fn(detection_model, input_tensor)
        
        # Extract obstacles and line boxes from detections
        obstacles, class_names, line_boxes, _ = extract_obstacles_from_detections(
            detections, category_index, height, width)
        
        # Calculate path planning
        clear_paths = get_clear_paths(width, obstacles)
        center_clearance = get_central_clearance(height, width, obstacles)
        
        # Determine if the path is safe - check for obstacles in line path first
        # In detection_thread:
        is_safe = check_path_safety(obstacles, line_boxes, clear_paths, center_clearance, height, width)
        
        # Update the global safety flags
        with safety_lock:
            path_is_safe = is_safe
            if not is_safe and prev_safe_state:
                # Path just became unsafe, trigger emergency stop
                emergency_stop = True
                print("⚠️ EMERGENCY STOP! Obstacle detected in line path")
            elif is_safe and not prev_safe_state:
                # Path just became safe again
                emergency_stop = False
                print("✓ Path clear, movement can resume")
        
        prev_safe_state = is_safe
        
        # Create combined visualization
        combined_img = visualize_combined(
            img_rgb, obstacles, class_names, line_boxes, clear_paths, center_clearance, 
            is_safe, show_paths, show_boxes, current_line_height
        )
        
        # Display the combined image
        cv2.imshow("Object Detection & Path Planning", combined_img)
        
        # Check for keyboard input
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("User requested exit")
            with safety_lock:
                detection_running = False
                global movement_running
                movement_running = False
            break
        elif key == ord('e'):
            show_paths = not show_paths
            print(f"Path visualization: {'ON' if show_paths else 'OFF'}")
        elif key == ord('r'):
            show_boxes = not show_boxes
            print(f"Bounding box visualization: {'ON' if show_boxes else 'OFF'}")
    
    print("Detection thread exiting...")
    cv2.destroyAllWindows()

def partial_move_forward(distance, step_size=0.5, speed=MOVEMENT_SPEED):
    """
    Move forward in small increments to allow for emergency stops.
    
    Args:
        distance: Total distance to move
        step_size: Size of each step in distance units
        speed: Motor speed
        
    Returns:
        Completed distance
    """
    global emergency_stop, movement_in_progress, movement_progress
    
    steps = max(1, int(distance / step_size))
    step_distance = distance / steps
    completed_distance = 0
    
    for i in range(steps):
        # Check for emergency stop
        with safety_lock:
            if emergency_stop:
                print(f"Movement interrupted after moving {completed_distance:.2f} units")
                return completed_distance
        
        # Update progress
        with movement_state_lock:
            movement_progress = completed_distance / distance
        
        # Move a small distance
        try:
            move_forward(step_distance, speed)
        except Exception as e:
            print(f"Error during forward movement: {e}")
            return completed_distance
        
        completed_distance += step_distance
        
        # Short delay to check for obstacles
        time.sleep(0.05)
    
    # Final progress update
    with movement_state_lock:
        movement_progress = 1.0
    
    return distance

def partial_turn_right(angle, step_size=10, speed=MOVEMENT_SPEED/3):
    """
    Turn right in small increments to allow for emergency stops.
    
    Args:
        angle: Total angle to turn
        step_size: Size of each step in angle units
        speed: Motor speed
        
    Returns:
        Completed angle
    """
    global emergency_stop, movement_in_progress, movement_progress
    
    steps = max(1, int(angle / step_size))
    step_angle = angle / steps
    completed_angle = 0
    
    for i in range(steps):
        # Check for emergency stop
        with safety_lock:
            if emergency_stop:
                print(f"Turn interrupted after turning {completed_angle:.2f} degrees")
                return completed_angle
        
        # Update progress
        with movement_state_lock:
            movement_progress = completed_angle / angle
        
        # Turn a small angle
        try:
            turn_right(step_angle, speed)
        except Exception as e:
            print(f"Error during turn: {e}")
            return completed_angle
        
        completed_angle += step_angle
        
        # Short delay to check for obstacles
        time.sleep(0.05)
    
    # Final progress update
    with movement_state_lock:
        movement_progress = 1.0
    
    return angle

def execute_movement(action, value):
    """
    Execute a single movement with support for emergency stop and resume.
    
    Args:
        action: Type of movement ("forward" or "turn_right")
        value: Distance or angle value
        
    Returns:
        Tuple of (completed, remaining) indicating progress
    """
    global emergency_stop, current_action, current_value, movement_in_progress
    
    with movement_state_lock:
        current_action = action
        current_value = value
        movement_in_progress = True
    
    if action == "forward":
        completed = partial_move_forward(value)
        remaining = value - completed
    elif action == "turn_right":
        completed = partial_turn_right(value)
        remaining = value - completed
    else:
        print(f"Unknown movement action: {action}")
        completed = 0
        remaining = value
    
    # Check if movement was interrupted by emergency stop
    with safety_lock:
        was_emergency = emergency_stop
    
    if was_emergency:
        # If emergency stop occurred, we'll need to resume later
        with movement_state_lock:
            movement_in_progress = False
        return completed, remaining
    else:
        # Completed successfully
        with movement_state_lock:
            movement_in_progress = False
        return value, 0

def movement_thread():
    """Thread for executing the rectangle movement pattern with obstacle avoidance"""
    global path_is_safe, emergency_stop, movement_running, current_movement_index
    
    print("Starting movement thread...")
    
    # Initialize robot for movement
    init_robot(robot)
    
    # Movement sequence for a rectangle
    movement_sequence = [
        ("forward", LONG_DISTANCE),
        ("turn_right", TURN_ANGLE),
        ("forward", SHORT_DISTANCE),
        ("turn_right", TURN_ANGLE),
        ("forward", LONG_DISTANCE),
        ("turn_right", TURN_ANGLE),
        ("forward", SHORT_DISTANCE),
        ("turn_right", TURN_ANGLE)
    ]
    
    print("\nStarting rectangle movement pattern with obstacle avoidance")
    if current_movement_index < len(movement_sequence):
        print("Current movement: {} {}".format(
            movement_sequence[current_movement_index][0], 
            movement_sequence[current_movement_index][1]
        ))
    
    # Track partially completed movements
    remaining_movements = movement_sequence.copy()
    partial_completed = 0
    
    while movement_running and current_movement_index < len(movement_sequence) and robot.step(TIME_STEP) != -1:
        # Check if path is safe to proceed
        local_path_is_safe = False
        local_emergency_stop = False
        
        with safety_lock:
            local_path_is_safe = path_is_safe
            local_emergency_stop = emergency_stop
        
        if local_path_is_safe and not local_emergency_stop:
            # Get current movement
            if current_movement_index < len(remaining_movements):
                action, total_value = remaining_movements[current_movement_index]
                value_to_do = total_value - partial_completed
                
                print(f"Executing {action} {value_to_do:.2f}/{total_value} units")
                
                # Execute the movement, which can be interrupted
                completed, remaining = execute_movement(action, value_to_do)
                
                if remaining > 0:
                    # Movement was interrupted, store progress
                    partial_completed += completed
                    print(f"Movement interrupted. Completed {partial_completed:.2f}/{total_value} of {action}")
                else:
                    # Movement completed fully
                    current_movement_index += 1
                    partial_completed = 0
                    
                    if current_movement_index < len(movement_sequence):
                        next_action, next_value = movement_sequence[current_movement_index]
                        print(f"Next movement: {next_action} {next_value}")
                    else:
                        print("Rectangle movement pattern completed!")
        else:
            # Path is blocked, wait
            if local_emergency_stop:
                # Just wait until the path is clear again
                time.sleep(0.1)
            else:
                # Small delay to prevent CPU hogging
                time.sleep(0.05)
    
    print("Movement thread exiting...")
    with safety_lock:
        movement_running = False

def main():
    global robot, detection_running, movement_running
    
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
    
    print("\nStarting threaded object detection and rectangle movement controller.")
    print("Controls:")
    print("  Press 'e' to toggle path visualization")
    print("  Press 'r' to toggle bounding box visualization")
    print("  Press 'q' to exit")
    
    try:
        # Create and start detection thread
        detection_thread_obj = threading.Thread(
            target=detection_thread,
            args=(detection_model, category_index, camera, width, height)
        )
        detection_thread_obj.daemon = True
        detection_thread_obj.start()
        
        # Create and start movement thread
        movement_thread_obj = threading.Thread(
            target=movement_thread
        )
        movement_thread_obj.daemon = True
        movement_thread_obj.start()
        
        # Wait for threads to finish
        while detection_running or movement_running:
            # Keep main thread alive but don't consume CPU
            time.sleep(0.1)
            
            # Check if robot connection is still active
            if robot.step(TIME_STEP) == -1:
                print("Robot connection lost, exiting...")
                detection_running = False
                movement_running = False
                break
        
        # Wait for threads to finish
        if detection_thread_obj.is_alive():
            detection_thread_obj.join(timeout=2.0)
        if movement_thread_obj.is_alive():
            movement_thread_obj.join(timeout=2.0)
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure everything is cleaned up
        detection_running = False
        movement_running = False
        stop()  # Make sure robot stops
        cv2.destroyAllWindows()
        print("Exit complete.")

if __name__ == "__main__":
    main()