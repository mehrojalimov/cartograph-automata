import json
import cv2
import numpy as np

# Load JSON file with labeled paths and obstacles
def load_labels(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Extract key elements from JSON
def extract_elements(data):
    line = []
    obstacles = []
    red_box = None

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        if label == "line":
            line = points
        elif label == "person" or label == "shelf":
            obstacles.append({'label': label, 'bbox': points})
        elif label == "box":  # Red hazard box
            red_box = points

    return line, obstacles, red_box

# Detect obstacles along the main path
def detect_obstacles(line, obstacles):
    detected = []
    for obstacle in obstacles:
        for point in line:
            x, y = point
            x_min, y_min = np.array(obstacle['bbox']).min(axis=0)
            x_max, y_max = np.array(obstacle['bbox']).max(axis=0)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                detected.append(obstacle)
                break
    return detected

# Generate alternative paths avoiding both the obstacle and the red box
def generate_alternate_paths(line, obstacles, red_box):
    if not obstacles:
        print("No obstacle detected.")
        return None  

    paths = {
        "left": [],
        "right": [],
        "curved": []
    }

    shift_x = 40  # Shift left or right
    shift_y = 30  # Shift for curved paths

    red_x_min, red_y_min, red_x_max, red_y_max = (None, None, None, None)
    if red_box:
        red_bbox = np.array(red_box)
        red_x_min, red_y_min = red_bbox.min(axis=0)
        red_x_max, red_y_max = red_bbox.max(axis=0)

    for (x, y) in line:
        # Check if the point is inside the obstacle
        inside_obstacle = False
        for obstacle in obstacles:
            x_min, y_min = np.array(obstacle['bbox']).min(axis=0)
            x_max, y_max = np.array(obstacle['bbox']).max(axis=0)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                inside_obstacle = True
                break

        if inside_obstacle:
            left_x, right_x = x - shift_x, x + shift_x

            # Ensure left detour does not enter the red box
            if red_box and (red_x_min <= left_x <= red_x_max and red_y_min <= y <= red_y_max):
                left_x -= 20  

            # Ensure right detour does not enter the red box
            if red_box and (red_x_min <= right_x <= red_x_max and red_y_min <= y <= red_y_max):
                right_x += 20  

            paths["left"].append([left_x, y])
            paths["right"].append([right_x, y])
            paths["curved"].append([x - shift_x // 2, y - shift_y if y > 100 else y + shift_y])  # Curved detour
        else:
            paths["left"].append([x, y])
            paths["right"].append([x, y])
            paths["curved"].append([x, y])

    return paths

# Select the best alternative path based on clearance
def choose_best_path(paths, obstacles, red_box, original_line):
    if paths is None:
        print("No obstacles detected. Using the original path.")
        return original_line  # Return the original path instead of None

    for key, path in paths.items():
        clear = True
        for (x, y) in path:
            for obstacle in obstacles:
                x_min, y_min = np.array(obstacle['bbox']).min(axis=0)
                x_max, y_max = np.array(obstacle['bbox']).max(axis=0)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    clear = False
                    break

            if red_box:
                red_x_min, red_y_min = np.array(red_box).min(axis=0)
                red_x_max, red_y_max = np.array(red_box).max(axis=0)
                if red_x_min <= x <= red_x_max and red_y_min <= y <= red_y_max:
                    clear = False
                    break

        if clear:
            print(f"Selected best path: {key}")
            return path

    print("No clear path found, defaulting to the original line.")
    return original_line

# Visualization function
def visualize_paths(image_path, line, obstacles, red_box, best_path):
    img = cv2.imread(image_path)

    # Draw original path (Green)
    for i in range(len(line) - 1):
        cv2.line(img, tuple(map(int, line[i])), tuple(map(int, line[i + 1])), (0, 255, 0), 2)

    # Draw obstacles (Red)
    for obstacle in obstacles:
        bbox = np.array(obstacle['bbox'])
        x_min, y_min = bbox.min(axis=0)
        x_max, y_max = bbox.max(axis=0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(img, obstacle['label'], (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw red hazard box
    if red_box:
        red_bbox = np.array(red_box)
        x_min, y_min = red_bbox.min(axis=0)
        x_max, y_max = red_bbox.max(axis=0)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 3)

    # Draw best alternative path (Blue)
    for i in range(len(best_path) - 1):
        cv2.line(img, tuple(map(int, best_path[i])), tuple(map(int, best_path[i + 1])), (255, 0, 0), 2)

    cv2.imshow("Optimal Path Planning", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    json_path = "segmentation_image/seq1_center_20250228-174452.json"
    image_path = "segmentation_image/seq1_center_20250228-174452.png"

    # Load data
    data = load_labels(json_path)
    line, obstacles, red_box = extract_elements(data)
    detected_obstacles = detect_obstacles(line, obstacles)
    
    # Generate alternative paths if obstacles exist
    alternative_paths = generate_alternate_paths(line, detected_obstacles, red_box)

    # Choose the best path, ensuring a valid path is always returned
    best_path = choose_best_path(alternative_paths, detected_obstacles, red_box, line)

    # Visualize results
    visualize_paths(image_path, line, detected_obstacles, red_box, best_path)
