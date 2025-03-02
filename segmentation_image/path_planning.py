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
def get_central_clearance(height, width, obstacles):
    """Find vertical clearance from the bottom to the bottom part of the first red box in the center.
    
    If no obstacle is found, return the full height.
    """
    center_x = width // 2
    min_y_max = height  # Default to full height (no obstacle case)

    for bbox in obstacles:
        x_min, _, x_max, y_max = bbox.min(axis=0)[0], bbox.min(axis=0)[1], bbox.max(axis=0)[0], bbox.max(axis=0)[1]

        if x_min <= center_x <= x_max:  # If the obstacle covers the center x
            min_y_max = min(min_y_max, y_max)  # Find the closest obstacle from the bottom

    # If min_y_max was never updated, return full height (no obstacle in the center)
    return height - min_y_max if min_y_max != height else height
  # Correct distance from the bottom to the obstacle


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

    clear_paths = get_clear_paths(width, obstacles)
    center_clearance = get_central_clearance(height, width, obstacles)

    return clear_paths, center_clearance

# Example usage
json_path = "segmentation_image/test7.json"
image_path = "segmentation_image/test7.png"
output_path = "segmentation_image/output.png"

clear_paths, center_clearance = draw_lines(image_path, json_path, output_path)
print("Clear paths (Centered x-coordinates):", clear_paths)
print("Vertical clearance at center:", center_clearance)
