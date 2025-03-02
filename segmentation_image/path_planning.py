import json
import cv2
import numpy as np

def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_obstacles(label_data):
    obstacles = []
    for shape in label_data['shapes']:
        label = shape['label']
        points = shape['points']
        if label in ['person', 'shelf']:  # Exclude these from line generation
            obstacles.append(np.array(points, dtype=int))
    return obstacles

def is_line_blocked(x, obstacles):
    """Check if a vertical line at x intersects any obstacle."""
    for bbox in obstacles:
        x_min, _ = bbox.min(axis=0)
        x_max, _ = bbox.max(axis=0)
        if x_min <= x <= x_max:
            return True
    return False

def get_clear_paths(width, obstacles, step=10):
    """Identify leftmost and rightmost x-coordinates of clear paths, centered at 0."""
    half_width = width // 2
    clear_x = []
    in_clear_area = False
    left_boundary = None

    for x in range(0, width, step):
        if not is_line_blocked(x, obstacles):
            if not in_clear_area:
                left_boundary = x - half_width  # Convert to centered coordinates
                in_clear_area = True
        else:
            if in_clear_area:
                clear_x.append((left_boundary, (x - step) - half_width))  # Convert to centered coordinates
                in_clear_area = False

    if in_clear_area:  # If last section was clear, close it
        clear_x.append((left_boundary, (width - 1) - half_width))

    return clear_x

def draw_lines(image_path, json_path, output_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    half_width = width // 2
    label_data = load_labels(json_path)
    obstacles = extract_obstacles(label_data)

    # Generate vertical lines
    line_spacing = 10  # Adjust density of lines
    for x in range(0, width, line_spacing):
        if not is_line_blocked(x, obstacles):
            centered_x = x - half_width  # Centering x-coordinate
            cv2.line(img, (x, 0), (x, height), (255, 0, 255), 2)  # Purple lines

    # Draw obstacle bounding boxes
    for bbox in obstacles:
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box
    
    cv2.imwrite(output_path, img)
    cv2.imshow("Segmented Lines", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return get_clear_paths(width, obstacles)

# Example usage
json_path = "segmentation_image/test3.json"
image_path = "segmentation_image/test3.png"
output_path = "segmentation_image/output.png"

clear_paths = draw_lines(image_path, json_path, output_path)
print("Clear paths (Centered x-coordinates):", clear_paths)
