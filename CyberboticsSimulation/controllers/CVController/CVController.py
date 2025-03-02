from controller import Robot, GPS
import math
from MovementFunctions import init_robot, turn_left, turn_right, move_forward, stop

def navigate_to_waypoint(robot, gps, goal_position):
    """
    Navigate the robot to a specific waypoint
    
    Args:
        robot: Webots robot instance
        gps: GPS device
        goal_position: Target waypoint coordinates [x, y, z]
    
    Returns:
        bool: True if waypoint was reached, False otherwise
    """
    TIME_STEP = 32
    DISTANCE_THRESHOLD = 0.1  # 10cm threshold to goal
    ANGLE_THRESHOLD = 5.0  # Continue if angle difference is less than this
    TURN_SCALE = 0.4  # Reduced turn scale to prevent overcorrection
    MIN_MOVEMENT_DISTANCE = 0.05  # Minimum distance to consider for heading calculation

    # Increase the angle threshold as we get closer to the goal
    CLOSE_DISTANCE = 1.5  # Distance at which we consider ourselves "close" to the goal
    CLOSE_ANGLE_THRESHOLD = 10.0  # More permissive angle threshold when close

    # Distance-based goal detection parameters
    DISTANCE_HISTORY_SIZE = 5  # How many distance measurements to keep
    DISTANCE_INCREASING_THRESHOLD = 4  # How many consecutive increases before we consider we've passed the goal

    # Movement history for better heading estimation
    position_history = []
    MAX_HISTORY = 5  # Keep track of last 5 positions

    # Distance history for goal detection
    distance_history = []

    # Variables to detect and handle movement problems
    consecutive_no_heading = 0
    MAX_NO_HEADING_COUNT = 3
    stuck_counter = 0
    MAX_STUCK_COUNT = 5
    distance_threshold_for_stuck = 0.03  # If we haven't moved at least this much, we might be stuck

    while robot.step(TIME_STEP) != -1:
        # Get current position
        current_position = gps.getValues()
        
        # Add current position to history
        position_history.append(current_position)
        if len(position_history) > MAX_HISTORY:
            position_history.pop(0)  # Remove oldest position
        
        # Compute distance to goal
        distance_to_goal = calculate_distance(current_position, goal_position)
        
        # Update distance history for goal detection
        distance_history.append(distance_to_goal)
        if len(distance_history) > DISTANCE_HISTORY_SIZE:
            distance_history.pop(0)
        
        # Stop if the robot is close enough to the goal
        if distance_to_goal < DISTANCE_THRESHOLD:
            print(f"🏁 Goal reached! Final position: {current_position}")
            stop()
            return True
        
        # Alternative goal detection: if we've passed the goal (distance is increasing)
        if is_distance_increasing(distance_history, DISTANCE_INCREASING_THRESHOLD) and distance_to_goal < 1.0:
            print(f"🏁 Goal likely passed or reached! Distance is now increasing. Final position: {current_position}")
            stop()
            return True
        
        # Check if we're stuck
        if len(distance_history) > 1 and abs(distance_history[-1] - distance_history[-2]) < distance_threshold_for_stuck:
            stuck_counter += 1
        else:
            stuck_counter = 0
        
        # If stuck for too long, make a decisive move to break out
        if stuck_counter >= MAX_STUCK_COUNT:
            print(f"⚠️ Robot seems stuck. Making a decisive move forward.")
            move_forward(0.2)  # Make a larger movement
            stuck_counter = 0
            robot.step(TIME_STEP * 2)  # Give time to move
            continue
        
        # Get heading from position history
        current_heading = calculate_heading_from_history(position_history, MIN_MOVEMENT_DISTANCE)
        
        # If we don't have a heading yet, move forward to establish one
        if current_heading is None:
            consecutive_no_heading += 1
            
            # If we've had too many consecutive no-heading frames, make a larger move
            if consecutive_no_heading > MAX_NO_HEADING_COUNT:
                print("⚠️ Multiple heading failures. Making a larger move forward.")
                move_forward(0.2)
                consecutive_no_heading = 0
            else:
                print("🔄 No heading available. Moving slightly to establish a heading.")
                move_forward(0.1)  # Move forward a small amount to establish heading
                
            robot.step(TIME_STEP)  # Give time to move
            continue
        else:
            consecutive_no_heading = 0  # Reset counter when we have a heading
        
        # Compute goal heading
        goal_heading = math.degrees(math.atan2(
            goal_position[2] - current_position[2],
            goal_position[0] - current_position[0]
        ))
        
        # Compute the angle difference and normalize
        angle_to_turn = normalize_angle(goal_heading - current_heading)
        
        # Get dynamic angle threshold based on distance
        current_angle_threshold = get_dynamic_angle_threshold(distance_to_goal, ANGLE_THRESHOLD, CLOSE_ANGLE_THRESHOLD, CLOSE_DISTANCE)
        
        print(f"📍 Position: ({current_position[0]:.2f}, {current_position[2]:.2f}) | " 
              f"Distance to Goal: {distance_to_goal:.2f}m | Current Heading: {current_heading:.2f}° | "
              f"Goal Heading: {goal_heading:.2f}° | Angle to Turn: {angle_to_turn:.2f}° | Threshold: {current_angle_threshold:.2f}°")
        
        # Check if we need a big turn (target is behind us)
        if needs_big_turn(angle_to_turn):
            turn_amount = 30  # Make a decisive turn when target is behind
            if angle_to_turn > 0:
                print(f"🔄 Big turn needed! Turning left {turn_amount} degrees")
                turn_left(turn_amount)
            else:
                print(f"🔄 Big turn needed! Turning right {turn_amount} degrees")
                turn_right(turn_amount)
            robot.step(TIME_STEP)
            continue
        
        # If angle is less than threshold, continue moving forward
        if abs(angle_to_turn) < current_angle_threshold:
            # When very close to goal, make smaller movements
            if distance_to_goal < 0.5:
                move_distance = min(distance_to_goal, 0.1)  # Smaller steps when close
                print(f"✔️ Close to goal. Making small movement forward ({move_distance:.2f}m).")
            else:
                move_distance = min(distance_to_goal, 0.5)  # Normal steps otherwise
                print(f"✔️ Aligned with goal (within threshold). Moving forward ({move_distance:.2f}m).")
                
            move_forward(move_distance)
        else:
            # Avoid excessive turning when close to goal
            if distance_to_goal < 0.5:
                turn_amount = max(min(abs(angle_to_turn) * 0.3, 10), 2)  # Gentler turning when close
            else:
                turn_amount = max(min(abs(angle_to_turn) * TURN_SCALE, 15), 3)  # Normal turning otherwise
                
            if angle_to_turn > 0:
                print(f"🔄 Turning left {turn_amount:.2f} degrees")
                turn_left(turn_amount)
            else:
                print(f"🔄 Turning right {turn_amount:.2f} degrees")
                turn_right(turn_amount)
        
        # Wait briefly after action to stabilize and get new GPS readings
        robot.step(TIME_STEP)
    
    return False

def calculate_heading_from_history(position_history, min_movement_distance):
    """Calculate heading based on multiple previous positions for better accuracy"""
    if len(position_history) < 2:
        return None
    
    # Use the first and last points in history for better estimation
    start = position_history[0]
    end = position_history[-1]
    
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    
    # Only calculate if there's significant movement
    distance_moved = math.sqrt(dx**2 + dz**2)
    if distance_moved < min_movement_distance:
        return None
        
    return math.degrees(math.atan2(dz, dx))

def normalize_angle(angle):
    """Normalize angle to range [-180, 180] degrees"""
    return (angle + 180) % 360 - 180

def calculate_distance(start, end):
    """Compute Euclidean distance in 2D (ignoring Y)"""
    dx = end[0] - start[0]
    dz = end[2] - start[2]
    return math.sqrt(dx**2 + dz**2)

def is_distance_increasing(distance_history, threshold):
    """Check if distance to goal is consistently increasing (we've passed the goal)"""
    if len(distance_history) < threshold + 1:
        return False
    
    # Check the last few measurements to see if distance is consistently increasing
    increasing_count = 0
    for i in range(1, threshold + 1):
        if distance_history[-i] > distance_history[-i-1]:
            increasing_count += 1
    
    return increasing_count >= threshold

def get_dynamic_angle_threshold(distance, normal_threshold, close_threshold, close_distance):
    """Return a dynamic angle threshold based on distance to goal"""
    if distance < close_distance:
        # Linear interpolation: as distance approaches 0, threshold approaches close_threshold
        return normal_threshold + (close_threshold - normal_threshold) * (1 - distance/close_distance)
    return normal_threshold

def needs_big_turn(angle_diff):
    """Determine if we need a big turn (target is behind us)"""
    return abs(angle_diff) > 90  # If goal is more than 90 degrees off our heading

def main():
    # Define waypoints
    WAYPOINTS = [
        [7.36, 1.18, -0.0105321],    # Waypoint 1
        [6.85, -9.55, -0.0101856],   # Waypoint 2
        [-7.54, -9.06, -0.000269982], # Waypoint 3
        [-7.02, 1.69, -0.000623052]  # Waypoint 4
    ]
    
    # Initialize Webots robot
    robot = Robot()
    
    # Initialize movement functions
    init_robot(robot)
    
    # Get GPS device
    gps = robot.getDevice("gps")
    gps.enable(32)  # Using TIME_STEP value of 32
    
    # Navigate to each waypoint in order
    for i, waypoint in enumerate(WAYPOINTS):
        print(f"\n🚶 Navigating to Waypoint {i+1}: {waypoint}")
        success = navigate_to_waypoint(robot, gps, waypoint)
        
        if not success:
            print(f"❌ Failed to reach Waypoint {i+1}")
            break
        
        print(f"✅ Successfully reached Waypoint {i+1}")
        stop(1)
        turn_right(80)
    
    print("🏁 Navigation complete!")
    stop()

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()