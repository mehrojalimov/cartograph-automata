import json
import cv2
import numpy as np
import os

# Load JSON file
def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Extract objects from the JSON file
def extract_objects(label_data):
    line = []
    obstacles = []
    
    for shape in label_data['shapes']:
        label = shape['label']
        points = shape['points']

        if label == 'line':
            line = points  # Robot path
        else:
            obstacles.append({'label': label, 'bbox': points})  # Store obstacles
    
    return line, obstacles

# Check if any obstacle is blocking the path
def is_obstacle_in_path(line, obstacles):
    for obstacle in obstacles:
        bbox = np.array(obstacle['bbox'])
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)

        for point in line:
            x, y = point
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return obstacle  # Obstacle detected
    return None

# Generate a new path to avoid obstacles
def find_alternate_path(line, obstacle):
    new_path = []
    bbox = np.array(obstacle['bbox'])
    x_min, y_min = bbox.min(axis=0)
    x_max, y_max = bbox.max(axis=0)
    
    # Move the path slightly left or right to avoid obstacle
    for x, y in line:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            new_path.append([x - 30, y])  # Try shifting left
        else:
            new_path.append([x, y])  # Keep the same path
    
    return new_path

# Draw objects and path on the image
def visualize_path(image_path, line, obstacles, new_path=None):
    img = cv2.imread(image_path)

    # Draw original path
    for i in range(len(line) - 1):
        cv2.line(img, tuple(map(int, line[i])), tuple(map(int, line[i + 1])), (0, 255, 0), 2)
    
    # Draw obstacles
    for obstacle in obstacles:
        bbox = np.array(obstacle['bbox'])
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(img, obstacle['label'], (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw new path if modified
    if new_path:
        for i in range(len(new_path) - 1):
            cv2.line(img, tuple(map(int, new_path[i])), tuple(map(int, new_path[i + 1])), (255, 0, 0), 2)

    cv2.imshow("Path Planning", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    json_path = "segmentation_image/seq1_center_20250228-174452.json"
    image_path = "segmentation_image/seq1_center_20250228-174452.png"

    label_data = load_labels(json_path)
    line, obstacles = extract_objects(label_data)
    
    obstacle = is_obstacle_in_path(line, obstacles)
    new_path = None

    if obstacle:
        print(f"Obstacle detected: {obstacle['label']}")
        new_path = find_alternate_path(line, obstacle)
    else:
        print("No obstacle detected.")

    visualize_path(image_path, line, obstacles, new_path)
