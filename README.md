
# Product link: https://sadakapp.streamlit.app/

# Traffic Flow Optimization and Congestion Management Tool - S.A.D.A.K

This repository contains our solution for the Karnataka State Police Hackathon: "Datathon"
![App Screenshot](cover.jpeg)
## Problem

Urban traffic congestion leads to economic losses, environmental impact, and decreased quality of life. Traditional traffic management methods are often limited in effectiveness.  

Problem Statement 1 - Evaluating the traffic congestion shown by map engine services against the calculated actual congestion using the Drone.

## Solution

The proposed solution is an object detection-based system using computer vision components.We leverage pre-trained models like YOLOv8 and supervision.These models can automatically detect and geo-reference bottlenecks and road congestions from images captured by the onboard cameras in real time, providing actionable insights for smooth and optimized traffic flow.

![image](https://github.com/Dev-on-go/S.A.D.A.K/assets/120119971/120599b9-5b57-4677-924e-7a0a3f4d21ef)
*Several Modules in this proposed solution still need to be worked upon

## Screenshots

### Showing Real Time annotation for Traffic Density Estimation and congestion Evaluation
Videos are divided into further subclips which are used for dataset creation.  
We have limited the annotations to vehicles to limit the chances of False positives, this can be improved when scaled further.
![image](https://github.com/Dev-on-go/S.A.D.A.K/assets/120119971/c81c8850-1a2f-4778-9b2e-720754362698)

### Marking of potential encroachment areas to identify posdible bottlenecks like parking in nearby junction areas

Only the vehicles inside the marked areas are tracked, and a wait-time period is generated for them. If a vehicles stays within the marked area for longer than the permissible duration an alert is genrated for communicationg the control room.
![image2](https://github.com/that-coding-kid/S.A.D.A.K/assets/120119971/04693fdb-9e43-4186-9f35-3b7d11ee3382)

