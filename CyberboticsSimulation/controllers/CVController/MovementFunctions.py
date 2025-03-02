#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Robot Movement Library
A collection of easy-to-use movement functions for the four-wheeled robot in Webots.
Includes angle normalization to correct turning discrepancies.

Author: Rokawoo
"""

import math

# Default configuration
TIME_STEP = 32
WHEEL_RADIUS = 0.1  # Approximate wheel radius in meters
ROBOT_WIDTH = 0.2   # Approximate distance between left and right wheels
DEFAULT_SPEED = 5  # Default speed for movement

# Correction factor for angle calculations
# This is the ratio between desired angle and actual turning angle
# 90 degrees (1.5708 rad) desired resulted in 0.518768 rad actual
# So the correction factor is approximately 3.03
ANGLE_CORRECTION_FACTOR = 3.15  # ≈ 3.03

# Global robot and motor references
robot = None
front_left_motor = None
front_right_motor = None
rear_left_motor = None
rear_right_motor = None


def init_robot(robot_instance):
    """Initialize the robot and its motors for movement."""
    global robot, front_left_motor, front_right_motor, rear_left_motor, rear_right_motor
    
    robot = robot_instance
    
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
    stop()


def stop(duration_seconds=0):
    """
    Stop all motors and optionally wait for a specified duration.
    
    Args:
        duration_seconds: Optional time to wait in seconds after stopping
    """
    # First stop all motors
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)
    
    # If a wait duration is specified, advance the simulation
    if duration_seconds > 0 and robot is not None:
        # Calculate number of time steps needed
        steps_needed = int((duration_seconds * 1000) / TIME_STEP)
        
        # Run the robot for the calculated number of steps
        for _ in range(steps_needed):
            robot.step(TIME_STEP)


def normalize_angle(degrees):
    """
    Normalize a desired turning angle in degrees to account for real-world calibration.
    
    Args:
        degrees: The desired turning angle in degrees
        
    Returns:
        The adjusted angle in degrees to achieve the desired turn
    """
    return degrees * ANGLE_CORRECTION_FACTOR


def turn_left(degrees, speed=DEFAULT_SPEED):
    """
    Turn the robot left by the specified number of degrees.
    
    Args:
        degrees: Angle to rotate in degrees (positive)
        speed: Motor speed for rotation (default: 1.0)
    """
    # Validate robot initialization
    if robot is None:
        raise RuntimeError("Robot not initialized. Call init_robot() first.")
    
    # Convert to positive value for left turn
    degrees = abs(degrees)
    
    # Apply normalization to correct the turning angle
    normalized_degrees = normalize_angle(degrees)
    angle_radians = math.radians(normalized_degrees)
    
    # Calculate time needed for rotation
    rotation_time = angle_radians * (ROBOT_WIDTH/2) / (WHEEL_RADIUS * speed)
    rotation_time_ms = int(rotation_time * 1000)  # Convert to milliseconds
    
    # Set motor velocities for left rotation
    front_left_motor.setVelocity(-speed)
    front_right_motor.setVelocity(speed)
    rear_left_motor.setVelocity(-speed)
    rear_right_motor.setVelocity(speed)
    
    # Calculate number of time steps needed
    steps_needed = max(1, rotation_time_ms // TIME_STEP)
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    stop()


def turn_right(degrees, speed=DEFAULT_SPEED):
    """
    Turn the robot right by the specified number of degrees.
    
    Args:
        degrees: Angle to rotate in degrees (positive)
        speed: Motor speed for rotation (default: 1.0)
    """
    # Validate robot initialization
    if robot is None:
        raise RuntimeError("Robot not initialized. Call init_robot() first.")
    
    # Convert to positive value for right turn
    degrees = abs(degrees)
    
    # Apply normalization to correct the turning angle
    normalized_degrees = normalize_angle(degrees)
    angle_radians = math.radians(normalized_degrees)
    
    # Calculate time needed for rotation
    rotation_time = angle_radians * (ROBOT_WIDTH/2) / (WHEEL_RADIUS * speed)
    rotation_time_ms = int(rotation_time * 1000)  # Convert to milliseconds
    
    # Set motor velocities for right rotation
    front_left_motor.setVelocity(speed)
    front_right_motor.setVelocity(-speed)
    rear_left_motor.setVelocity(speed)
    rear_right_motor.setVelocity(-speed)
    
    # Calculate number of time steps needed
    steps_needed = max(1, rotation_time_ms // TIME_STEP)
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    stop()


def move_forward(distance, speed=DEFAULT_SPEED * 3):
    """
    Move the robot forward by the specified distance in meters.
    
    Args:
        distance: Distance to move in meters (positive)
        speed: Motor speed for movement (default: 1.0)
    """
    # Validate robot initialization
    if robot is None:
        raise RuntimeError("Robot not initialized. Call init_robot() first.")
    
    # Convert to positive value for forward movement
    distance = abs(distance)
    
    # Calculate time needed for movement
    movement_time = distance / (WHEEL_RADIUS * speed)
    movement_time_ms = int(movement_time * 1000)  # Convert to milliseconds
    
    # Set motor velocities for forward movement
    front_left_motor.setVelocity(speed)
    front_right_motor.setVelocity(speed)
    rear_left_motor.setVelocity(speed)
    rear_right_motor.setVelocity(speed)
    
    # Calculate number of time steps needed
    steps_needed = max(1, movement_time_ms // TIME_STEP)
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    stop()


def move_backward(distance, speed=DEFAULT_SPEED * 3):
    """
    Move the robot backward by the specified distance in meters.
    
    Args:
        distance: Distance to move in meters (positive)
        speed: Motor speed for movement (default: 1.0)
    """
    # Validate robot initialization
    if robot is None:
        raise RuntimeError("Robot not initialized. Call init_robot() first.")
    
    # Convert to positive value for backward distance
    distance = abs(distance)
    
    # Calculate time needed for movement
    movement_time = distance / (WHEEL_RADIUS * speed)
    movement_time_ms = int(movement_time * 1000)  # Convert to milliseconds
    
    # Set motor velocities for backward movement (negative speed)
    front_left_motor.setVelocity(-speed)
    front_right_motor.setVelocity(-speed)
    rear_left_motor.setVelocity(-speed)
    rear_right_motor.setVelocity(-speed)
    
    # Calculate number of time steps needed
    steps_needed = max(1, movement_time_ms // TIME_STEP)
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    stop()


def rotate(degrees, speed=DEFAULT_SPEED):
    """
    Rotate the robot by the specified angle in degrees.
    Positive degrees = left turn, negative degrees = right turn.
    
    Args:
        degrees: Angle to rotate in degrees (positive for left, negative for right)
        speed: Motor speed for rotation (default: 1.0)
    """
    if degrees > 0:
        turn_left(degrees, speed)
    elif degrees < 0:
        turn_right(-degrees, speed)
    # If degrees is 0, do nothing


def calibrate_angle_correction(test_angle=90):
    """
    Utility function to calibrate the angle correction factor.
    
    This function should be used if you want to recalibrate the angle correction factor
    based on actual observed turning behavior. Call this with a known angle, then
    manually measure the actual rotation achieved and update the ANGLE_CORRECTION_FACTOR.
    
    Args:
        test_angle: The angle in degrees to use for calibration (default: 90)
    """
    if robot is None:
        raise RuntimeError("Robot not initialized. Call init_robot() first.")
    
    print(f"Calibration test: Turning {test_angle} degrees left...")
    print(f"Current correction factor: {ANGLE_CORRECTION_FACTOR}")
    
    # Calculate the actual angle in radians that would be used
    normalized_angle = normalize_angle(test_angle)
    print(f"With correction, using {normalized_angle} degrees ({math.radians(normalized_angle)} radians)")
    
    # Turn without normalization for calibration purposes
    angle_radians = math.radians(test_angle)
    
    # Calculate time needed for rotation
    rotation_time = angle_radians * (ROBOT_WIDTH/2) / (WHEEL_RADIUS * DEFAULT_SPEED)
    rotation_time_ms = int(rotation_time * 1000)
    
    # Set motor velocities for left rotation
    front_left_motor.setVelocity(DEFAULT_SPEED)
    front_right_motor.setVelocity(-DEFAULT_SPEED)
    rear_left_motor.setVelocity(DEFAULT_SPEED)
    rear_right_motor.setVelocity(-DEFAULT_SPEED)
    
    # Calculate number of time steps needed
    steps_needed = max(1, rotation_time_ms // TIME_STEP)
    
    # Run the robot for the calculated number of steps
    for _ in range(int(steps_needed)):
        robot.step(TIME_STEP)
    
    # Stop all motors
    stop()
    
    print("Calibration turn completed.")
    print("Measure the actual angle turned, then update the ANGLE_CORRECTION_FACTOR.")
    print(f"If the robot turned X degrees, the new factor should be approximately {test_angle}/X.")


if __name__ == "__main__":
    # This code will only run when the library is executed directly
    # It serves as a simple test of the movement functions
    try:
        # Import necessary modules
        from controller import Robot
        
        # Create a robot instance
        test_robot = Robot()
        print("Robot instance created for testing")
        
        # Initialize the robot
        init_robot(test_robot)
        print("Robot initialized")
        
        # Perform a simple left turn test with normalization
        print(f"Testing left turn movement (90 degrees with correction factor {ANGLE_CORRECTION_FACTOR})...")
        stop(1)
        move_forward(20.0)
        turn_left(90)
        turn_right(90)
        move_forward(1.0)
        move_backward(1.0)
        stop(1)
        print("Left turn completed")
        
        # Wait a bit
        for _ in range(20):
            test_robot.step(TIME_STEP)
        
        # Option to run the calibration procedure
        # Comment/uncomment this line to enable/disable calibration
        # calibrate_angle_correction(90)
        
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Test failed with error: {e}")