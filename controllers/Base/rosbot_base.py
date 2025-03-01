#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from controller import Robot, Motor, PositionSensor, Camera, RangeFinder, Lidar, Accelerometer, Gyro, Compass, DistanceSensor

TIME_STEP = 32
MAX_VELOCITY = 26

def main():
    # define variables
    # motors
    front_left_motor, front_right_motor = None, None
    rear_left_motor, rear_right_motor = None, None
    front_left_position_sensor, front_right_position_sensor = None, None
    rear_left_position_sensor, rear_right_position_sensor = None, None
    avoidance_speed = [0.0, 0.0]
    base_speed = 6.0
    motor_speed = [0.0, 0.0]
    # RGBD camera
    camera_rgb, camera_depth = None, None
    # rotational lidar
    lidar = None
    # IMU
    accelerometer, gyro, compass = None, None, None
    # distance sensors
    distance_sensors = [None, None, None, None]
    distance_sensors_value = [0.0, 0.0, 0.0, 0.0]

    # set empirical coefficients for collision avoidance
    coefficients = [[15.0, -9.0], [-15.0, 9.0]]

    # initialize Webots
    robot = Robot()

    # get a handler to the motors and set target position to infinity (speed control)
    front_left_motor = robot.getDevice("fl_wheel_joint")
    front_right_motor = robot.getDevice("fr_wheel_joint")
    rear_left_motor = robot.getDevice("rl_wheel_joint")
    rear_right_motor = robot.getDevice("rr_wheel_joint")
    front_left_motor.setPosition(float('inf'))
    front_right_motor.setPosition(float('inf'))
    rear_left_motor.setPosition(float('inf'))
    rear_right_motor.setPosition(float('inf'))
    front_left_motor.setVelocity(0.0)
    front_right_motor.setVelocity(0.0)
    rear_left_motor.setVelocity(0.0)
    rear_right_motor.setVelocity(0.0)

    # get a handler to the position sensors and enable them
    front_left_position_sensor = robot.getDevice("front left wheel motor sensor")
    front_right_position_sensor = robot.getDevice("front right wheel motor sensor")
    rear_left_position_sensor = robot.getDevice("rear left wheel motor sensor")
    rear_right_position_sensor = robot.getDevice("rear right wheel motor sensor")
    front_left_position_sensor.enable(TIME_STEP)
    front_right_position_sensor.enable(TIME_STEP)
    rear_left_position_sensor.enable(TIME_STEP)
    rear_right_position_sensor.enable(TIME_STEP)

    # get a handler to the ASTRA rgb and depth cameras and enable them
    camera_rgb = robot.getDevice("camera rgb")
    camera_depth = robot.getDevice("camera depth")
    camera_rgb.enable(TIME_STEP)
    camera_depth.enable(TIME_STEP)

    # get a handler to the RpLidarA2 and enable it
    lidar = robot.getDevice("laser")
    lidar.enable(TIME_STEP)
    lidar.enablePointCloud()

    # get a handler to the IMU devices and enable them
    accelerometer = robot.getDevice("imu accelerometer")
    gyro = robot.getDevice("imu gyro")
    compass = robot.getDevice("imu compass")
    accelerometer.enable(TIME_STEP)
    gyro.enable(TIME_STEP)
    compass.enable(TIME_STEP)

    # get a handler to the distance sensors and enable them
    distance_sensors[0] = robot.getDevice("fl_range")
    distance_sensors[1] = robot.getDevice("rl_range")
    distance_sensors[2] = robot.getDevice("fr_range")
    distance_sensors[3] = robot.getDevice("rr_range")
    distance_sensors[0].enable(TIME_STEP)
    distance_sensors[1].enable(TIME_STEP)
    distance_sensors[2].enable(TIME_STEP)
    distance_sensors[3].enable(TIME_STEP)

    # main loop
    while robot.step(TIME_STEP) != -1:
        # get accelerometer values
        a = accelerometer.getValues()
        print(f"accelerometer values = {a[0]:.2f} {a[1]:.2f} {a[2]:.2f}")

        # get distance sensors values
        for i in range(4):
            distance_sensors_value[i] = distance_sensors[i].getValue()

        # compute motors speed
        for i in range(2):
            avoidance_speed[i] = 0.0
            for j in range(1, 3):
                avoidance_speed[i] += (2.0 - distance_sensors_value[j]) * (2.0 - distance_sensors_value[j]) * coefficients[i][j - 1]
            motor_speed[i] = base_speed + avoidance_speed[i]
            motor_speed[i] = min(MAX_VELOCITY, motor_speed[i])

        # set speed values
        front_left_motor.setVelocity(motor_speed[0])
        front_right_motor.setVelocity(motor_speed[1])
        rear_left_motor.setVelocity(motor_speed[0])
        rear_right_motor.setVelocity(motor_speed[1])

if __name__ == "__main__":
    main()