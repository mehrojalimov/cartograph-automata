#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image and Label Collection Script for Computer Vision Training
This script finds and copies images with matching JSON label files from multiple source directories
to a single destination folder, avoiding duplicates and validating image formats.

Author: Rokawoo
"""

import os
import time
import math
from pathlib import Path
from controller import Robot

OUTPUT_FOLDER = "captured_images_[ATTRIBUTES]"  # Replace with your desired folder name

# Set OUTPUT_DIRECTORY to the desired path in CVHelperScripts
current_dir = Path(os.getcwd())
base_dir = current_dir.parents[2]  # Go back 3 folders
OUTPUT_DIRECTORY = os.path.join(base_dir, "CVHelperScripts", "captured_cv_training_data", OUTPUT_FOLDER)

# Create the directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
print(f"Using output directory: {OUTPUT_DIRECTORY}")

# User Configuration Variables
TIME_STEP = 32
ROTATION_DEGREES = 80  # Rotation angle in degrees
MOTOR_SPEED = 0.5  # Motor speed for rotation
FORWARD_SPEED = 2.0  # Speed for forward movement
FORWARD_DISTANCE = 4  # Meters to move forward in mode 2
OPERATION_MODE = 2  # 1: Take one set of pictures and exit 
                   # 2: Continuously take pictures, moving forward between sets
MAX_SEQUENCES = -1  # Maximum number of sequences to capture in mode 2 (set to -1 for unlimited)
WHEEL_RADIUS = 0.1  # Approximate wheel radius in meters
ROBOT_WIDTH = 0.2   # Approximate distance between left and right wheels

# Direction Configuration
# Set to True for directions you want to capture, False for directions to skip
CAPTURE_DIRECTIONS = {
    "center": True,  # Original forward-facing position
    "left": False,    # Left rotation
    "right": True,   # Right rotation
    "back": False,   # 180-degree rotation
    "up": False,     # Looking up (if robot has tilt capability)
    "down": False    # Looking down (if robot has tilt capability)
}


def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created directory: {OUTPUT_DIRECTORY}")
    
    # Initialize Webots robot
    robot = Robot()
    
    # Get camera and enable it
    camera = robot.getDevice("camera rgb")  # Using the RGB camera from your robot
    camera.enable(TIME_STEP)
    
    # Get wheel motors
    front_left_motor = robot.getDevice("fl_wheel_joint")
    front_right_motor = robot.getDevice("fr_wheel_joint")
    rear_left_motor = robot.getDevice("rl_wheel_joint")
    rear_right_motor = robot.getDevice("rr_wheel_joint")
    
    # Set position to infinity for velocity control
    front_left_motor.setPosition(float('inf'))
    front_right_motor.setPosition(float('inf'))
    rear_left_motor.setPosition(float('inf'))
    rear_right_motor.setPosition(float('inf'))
    
    # Initially stop all motors
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)
    
    # Print active directions
    active_dirs = [dir for dir, active in CAPTURE_DIRECTIONS.items() if active]
    print(f"Starting in mode {OPERATION_MODE}")
    print(f"Taking pictures in these directions: {', '.join(active_dirs)}")
    
    if OPERATION_MODE == 2 and MAX_SEQUENCES > 0:
        print(f"Will capture {MAX_SEQUENCES} sequences")
    elif OPERATION_MODE == 2:
        print("Will capture unlimited sequences (until program is terminated)")
    
    sequence_count = 0
    running = True
    
    # Main control loop
    try:
        while robot.step(TIME_STEP) != -1 and running:
            sequence_count += 1
            print(f"\nStarting sequence #{sequence_count}")
            current_rotation = 0  # Keep track of current rotation
            
            # Center position
            if CAPTURE_DIRECTIONS["center"]:
                print("Taking picture at center position...")
                robot.step(TIME_STEP)  # Short stabilization time
                take_picture(camera, f"seq{sequence_count}_center")
            
            # Left position
            if CAPTURE_DIRECTIONS["right"]:
                print("Rotating left...")
                rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, ROTATION_DEGREES)
                current_rotation += ROTATION_DEGREES
                
                print("Taking picture at left position...")
                robot.step(TIME_STEP)  # Short stabilization time
                take_picture(camera, f"seq{sequence_count}_left")
            
            # Return to center if needed for next position
            if current_rotation != 0 and (CAPTURE_DIRECTIONS["center"] or CAPTURE_DIRECTIONS["right"] or CAPTURE_DIRECTIONS["back"]):
                print("Returning to center...")
                rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, -current_rotation)
                current_rotation = 0
                robot.step(TIME_STEP)  # Short stabilization time
            
            # Right position
            if CAPTURE_DIRECTIONS["left"]:
                print("Rotating right...")
                rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, -ROTATION_DEGREES)
                current_rotation -= ROTATION_DEGREES
                
                print("Taking picture at right position...")
                robot.step(TIME_STEP)  # Short stabilization time
                take_picture(camera, f"seq{sequence_count}_right")
            
            # Back position (180 degree rotation)
            if CAPTURE_DIRECTIONS["back"]:
                # Return to center first if we're not already there
                if current_rotation != 0:
                    print("Returning to center...")
                    rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, -current_rotation)
                    current_rotation = 0
                    robot.step(TIME_STEP)
                
                print("Rotating to back position...")
                rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, 180)
                current_rotation = 180
                
                print("Taking picture at back position...")
                robot.step(TIME_STEP)
                take_picture(camera, f"seq{sequence_count}_back")
            
            # Return to center position before ending sequence
            if current_rotation != 0:
                print("Returning to center position...")
                rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, -current_rotation)
                robot.step(TIME_STEP)  # Short stabilization time
            
            # If mode 1, exit after one sequence
            if OPERATION_MODE == 1:
                print("Mode 1 complete. Exiting...")
                running = False
            
            # If mode 2, move forward a bit and repeat
            elif OPERATION_MODE == 2:
                # Check if we've reached max sequences
                if MAX_SEQUENCES > 0 and sequence_count >= MAX_SEQUENCES:
                    print(f"Reached maximum of {MAX_SEQUENCES} sequences. Exiting...")
                    running = False
                else:
                    print("Moving forward for next sequence...")
                    move_forward(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, FORWARD_DISTANCE)
                    print("Ready for next sequence.")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop all motors before exiting
        front_left_motor.setVelocity(0.0)
        front_right_motor.setVelocity(0.0)
        rear_left_motor.setVelocity(0.0)
        rear_right_motor.setVelocity(0.0)
        print("Motors stopped. Exit complete.")


def take_picture(camera, position_name):
    """Capture an image from the camera and save it to the output directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{OUTPUT_DIRECTORY}/{position_name}_{timestamp}.png"
    
    # Save the image
    camera.saveImage(filename, 100)  # Save at 100% quality
    print(f"Saved image: {filename}")


def rotate_robot(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, angle_degrees):
    """Rotate the robot by the specified angle in degrees."""
    # Convert degrees to radians
    angle_radians = math.radians(angle_degrees)
    
    # Calculate time needed for rotation based on speed and angle
    # Time needed = angle * (robot_width/2) / (wheel_radius * speed)
    rotation_time = abs(angle_radians) * (ROBOT_WIDTH/2) / (WHEEL_RADIUS * MOTOR_SPEED)
    rotation_time_ms = int(rotation_time * 1000)  # Convert to milliseconds
    
    # Set rotation direction based on angle sign
    direction = 1 if angle_degrees > 0 else -1
    
    # Set motor velocities for rotation (opposite directions for left/right wheels)
    front_left_motor.setVelocity(direction * MOTOR_SPEED)
    front_right_motor.setVelocity(-direction * MOTOR_SPEED)
    rear_left_motor.setVelocity(direction * MOTOR_SPEED)
    rear_right_motor.setVelocity(-direction * MOTOR_SPEED)
    
    # Calculate number of time steps needed
    steps_needed = rotation_time_ms // TIME_STEP
    
    # Make sure we rotate for at least one step
    steps_needed = max(1, steps_needed)
    
    print(f"Rotating {angle_degrees} degrees for {steps_needed} steps")
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)


def move_forward(robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor, distance):
    """Move the robot forward by the specified distance in meters."""
    # Calculate time needed for movement based on speed and distance
    # Time needed = distance / (wheel_radius * speed)
    movement_time = distance / (WHEEL_RADIUS * FORWARD_SPEED)
    movement_time_ms = int(movement_time * 1000)  # Convert to milliseconds
    
    # Set motor velocities for forward movement (all wheels same direction)
    front_left_motor.setVelocity(FORWARD_SPEED)
    front_right_motor.setVelocity(FORWARD_SPEED)
    rear_left_motor.setVelocity(FORWARD_SPEED)
    rear_right_motor.setVelocity(FORWARD_SPEED)
    
    # Calculate number of time steps needed
    steps_needed = movement_time_ms // TIME_STEP
    steps_needed = max(1, steps_needed)
    
    print(f"Moving forward {distance} meters for {steps_needed} steps")
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)


if __name__ == "__main__":
    main()