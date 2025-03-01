from controller import Robot, GPS
import math
from MovementFunctions import init_robot, turn_left, turn_right, move_forward, stop  # Import movement functions

TIME_STEP = 32
GOAL_POSITION = [-3.67, 1.18, -0.00293147]
ANGLE_THRESHOLD = 5   # Stop turning if within ±5 degrees
DISTANCE_THRESHOLD = 1  # Stop moving if within 10cm of the goal
TURN_SCALE = 0.5  # Proportional turning to prevent oscillations

# Initialize Webots robot
robot = Robot()

# Initialize movement functions
init_robot(robot)

# Get GPS device
gps = robot.getDevice("gps")
gps.enable(TIME_STEP)

# Initialize previous position (set to None initially)
previous_position = None

def calculate_heading(start, end):
    """ Estimate the heading angle in degrees using two GPS positions. """
    dx = end[0] - start[0]
    dz = end[2] - start[2]  # Webots uses X-Z plane
    return math.degrees(math.atan2(dz, dx)) if (dx != 0 or dz != 0) else None

def normalize_angle(angle):
    """ Normalize angle to range [-180, 180] degrees. """
    return (angle + 180) % 360 - 180  # Keep within [-180, 180] range

def calculate_distance(start, end):
    """ Compute Euclidean distance in 2D (ignoring Y). """
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    return math.sqrt(dx**2 + dz**2)

while robot.step(TIME_STEP) != -1:
    # Get current position
    current_position = gps.getValues()

    # If we don't have a previous position yet, set it and continue
    if previous_position is None:
        previous_position = current_position
        continue

    # Compute distance to goal
    distance_to_goal = calculate_distance(current_position, GOAL_POSITION)

    # Stop if the robot is close enough to the goal
    if distance_to_goal < DISTANCE_THRESHOLD:
        print("🏁 Goal reached! Stopping robot.")
        stop()
        break

    # Estimate the robot's current heading using GPS movement
    estimated_heading = calculate_heading(previous_position, current_position)
    
    # If the robot hasn't moved, keep turning to establish a heading
    if estimated_heading is None:
        print("🔄 No movement detected. Turning slightly to determine heading.")
        turn_left(5)
        continue

    # Compute goal heading
    goal_heading = calculate_heading(current_position, GOAL_POSITION)

    # Compute the angle difference and normalize
    angle_to_turn = normalize_angle(goal_heading - estimated_heading)

    print(f"📍 Distance to Goal: {distance_to_goal:.2f}m | Estimated Heading: {estimated_heading:.2f}° | Goal Heading: {goal_heading:.2f}° | Angle to Turn: {angle_to_turn:.2f}°")

    # If the robot is aligned with the goal, move forward
    if abs(angle_to_turn) < ANGLE_THRESHOLD:
        print("✔️ Aligned with goal. Moving forward.")
        move_forward(distance_to_goal)
    else:
        # Turn proportionally to prevent overshooting
        turn_amount = max(min(abs(angle_to_turn) * TURN_SCALE, 10), 2)  # Min 2°, Max 10°
        if angle_to_turn > 0:
            print(f"🔄 Turning left {turn_amount:.2f} degrees")
            turn_left(turn_amount)
        else:
            print(f"🔄 Turning right {turn_amount:.2f} degrees")
            turn_right(turn_amount)

    # Update previous position for the next step
    previous_position = current_position
