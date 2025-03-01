<div align="center">
  <img src="/api/placeholder/400/300" alt="Lorem Ipsum" align="center" width="400px"/>
  <h1>Lorem Ipsum</h1>
  <p>By Rokawoo, Koy, Arya, Mehroj</p>
</div>

> [!CAUTION]
> :star: Our Codefest Project is really Based!

## Abstract

Current AMR navigation relies heavily on basic sensor systems like LIDAR and ultrasonic sensors, which simply detect obstacles and follow predetermined rules without true environmental understanding. ____ bridges this gap by creating a hybrid system where computer vision not only detects obstacles but intelligently interprets the environment to make contextual decisions. 

By translating visual data into movement commands, our model enables robots to safely navigate around obstacles and people in real-time, even when encountering previously unseen scenarios. This proof-of-concept demonstrates how CV-enhanced navigation significantly improves safety and efficiency in both warehouse, public settings, and applications beyond, allowing AMRs to dynamically adapt to changing conditions while maintaining operational objectives.

The system's ability to decide when to deviate from programmed paths and when to return to them represents a meaningful step toward more autonomous and adaptable robotic systems for industrial and service applications.

## Demo

[Demo Video](#)

## Project Media

<div align="center">
  <img src="/api/placeholder/400/300" alt="Lorem Ipsum"/>
  <img src="/api/placeholder/400/300" alt="Lorem Ipsum"/>
  <img src="/api/placeholder/400/300" alt="Lorem Ipsum"/>
</div>

## :mag: Key Features

- **Intelligent Path Deviation**: Makes contextual decisions about when to leave programmed paths
- **Obstacle Classification**: Distinguishes between static obstacles, humans, and other mobile objects
- **Return-to-Path Algorithm**: Efficiently returns to optimal routes after obstacle avoidance
- **Simulation Validated**: Tested in various Webots environments including warehouses and public spaces
- **Hybrid Sensing Integration**: Combines traditional sensors with computer vision for robust navigation

## 🛠️ Libraries and Tools Used

- **TensorFlow Model Zoo**: For fine-tuning the SSD ResNet50 V1 FPN 640x640 (RetinaNet50) model
- **Webots**: Professional robot simulator for testing and validation
- **Python**: Primary programming language
- **OpenCV**: For additional image processing
