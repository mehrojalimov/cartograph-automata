"""
Multi-Goal Robot Navigation System

This module controls a robot to autonomously navigate through a sequence of predefined 
goal positions using GPS-based heading estimation and simple navigation algorithms.

The robot uses GPS to determine its position, calculates the heading to the current goal,
and adjusts its movement accordingly by turning or moving forward. When each goal is reached,
the robot automatically proceeds to the next goal in the sequence, looping back to the first
goal after reaching the final one.

Usage:
    Run this script directly with a properly configured Webots robot simulation
    that includes a GPS device named "gps".
"""

from controller import Robot, GPS
import math
from MovementFunctions import init_robot, turn_left, turn_right, move_forward, stop

# Constants
TIME_STEP = 32
GOAL_POSITIONS = [
    [6.84, 1.18, -0.0101735],            # Goal 1
    [6.84, -9.07, -0.0101776],           # Goal 2
    [-7.01, -9.07, -0.000634254],        # Goal 3
    [-7.02, 1.18, -0.000623302]          # Goal 4
]
ANGLE_THRESHOLD = 5   # Stop turning if within ±5 degrees
DISTANCE_THRESHOLD = 1  # Stop moving if within 1m of the goal
TURN_SCALE = 0.5  # Proportional turning to prevent oscillations
MIN_TURN_ANGLE = 2  # Minimum turn angle in degrees
MAX_TURN_ANGLE = 10  # Maximum turn angle in degrees

def calculate_heading(start, end):
    """ 
    Estimate the heading angle in degrees using two GPS positions.
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        
    Returns:
        float or None: Heading angle in degrees, None if no movement detected
    """
    dx = end[0] - start[0]
    dz = end[2] - start[2]  # Webots uses X-Z plane
    return math.degrees(math.atan2(dz, dx)) if (dx != 0 or dz != 0) else None

def normalize_angle(angle):
    """ 
    Normalize angle to range [-180, 180] degrees.
    
    Args:
        angle: Angle in degrees
        
    Returns:
        float: Normalized angle in degrees
    """
    return (angle + 180) % 360 - 180

def calculate_distance(start, end):
    """ 
    Compute Euclidean distance in 2D (ignoring Y).
    
    Args:
        start: Starting position [x, y, z]
        end: Ending position [x, y, z]
        
    Returns:
        float: Distance in meters
    """
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    return math.sqrt(dx**2 + dz**2)

def main():
    # Initialize Webots robot
    robot = Robot()
    
    # Initialize movement functions
    init_robot(robot)
    
    # Get GPS device
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
    
    # Initialize previous position and current goal index
    previous_position = None
    current_goal_index = 0
    
    # Main control loop
    while robot.step(TIME_STEP) != -1:
        # Get current position
        current_position = gps.getValues()
        
        # Get current goal position
        current_goal = GOAL_POSITIONS[current_goal_index]
        
        # If we don't have a previous position yet, set it and continue
        if previous_position is None:
            previous_position = current_position
            continue
        
        # Compute distance to current goal
        distance_to_goal = calculate_distance(current_position, current_goal)
        
        # Check if the robot has reached the current goal
        if distance_to_goal < DISTANCE_THRESHOLD:
            print(f"🏁 Goal {current_goal_index + 1} reached!")
            
            # Move to the next goal
            current_goal_index = (current_goal_index + 1) % len(GOAL_POSITIONS)
            next_goal = GOAL_POSITIONS[current_goal_index]
            
            print(f"🎯 Setting new target: Goal {current_goal_index + 1} at position {next_goal}")
            continue
        
        # Estimate the robot's current heading using GPS movement
        estimated_heading = calculate_heading(previous_position, current_position)
        
        # If the robot hasn't moved, turn slightly to establish a heading
        if estimated_heading is None:
            print("🔄 No movement detected. Turning slightly to determine heading.")
            turn_left(MIN_TURN_ANGLE)
            continue
        
        # Compute goal heading
        goal_heading = calculate_heading(current_position, current_goal)
        
        # Compute the angle difference and normalize
        angle_to_turn = normalize_angle(goal_heading - estimated_heading)
        
        print(f"📍 Goal {current_goal_index + 1} | Distance: {distance_to_goal:.2f}m | Heading: {estimated_heading:.2f}° | Goal Heading: {goal_heading:.2f}° | Angle to Turn: {angle_to_turn:.2f}°")
        
        # If the robot is aligned with the goal, move forward
        if abs(angle_to_turn) < ANGLE_THRESHOLD:
            print("✔️ Aligned with goal. Moving forward.")
            # Move a bit at a time to avoid overshooting
            move_distance = min(distance_to_goal, 2.0)  # Move up to 2 meters at a time
            move_forward(move_distance)
        else:
            # Calculate proportional turn amount with clamping
            turn_amount = min(max(abs(angle_to_turn) * TURN_SCALE, MIN_TURN_ANGLE), MAX_TURN_ANGLE)
            
            if angle_to_turn > 0:
                print(f"🔄 Turning left {turn_amount:.2f} degrees")
                turn_left(turn_amount)
            else:
                print(f"🔄 Turning right {turn_amount:.2f} degrees")
                turn_right(turn_amount)
        
        # Update previous position for the next step
        previous_position = current_position

if __name__ == "__main__":
    main()