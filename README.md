# PalsyTrack
Longitudinal Facial Asymmetry Monitor

## About the Project
I built this tool after learning about Bell's Palsy in class and wanting to apply computer vision to a real-world medical problem. In clinical settings, tracking facial weakness is often subjective or requires expensive 3D imaging. PalsyTrack offers a way to monitor recovery or symptom progression objectively using just a standard webcam. It quantifies facial symmetry and flags significant changes over time.
## How it Works
The software uses Google's MediaPipe Face Mesh to extract 478 3D facial landmarks in real-time. It doesn't just look at raw pixels; it understands the geometry of the face.

1. Consistency First: To ensure photos are comparable day-to-day, I built a "Ghost" overlay system. You align your face with a static guide (which you can customize to your face shape) before capturing. This minimizes errors caused by head rotation or camera distance.

2. Normalization: All measurements are scaled relative to your Inter-Pupillary Distance (IPD). This means sitting closer or further from the camera won't skew the data.

3. Statistical Analysis: Instead of just showing raw numbers, the system builds a personalized baseline from your first few scans. It uses Z-scores (standard deviations) to determine if a new change is statistically significant or just normal daily variation.
## Core Features

1. Precise Metric Tracking: Monitors specific biomarkers like Palpebral Fissure Height (eye opening), Oral Commissure Droop (mouth corners), and Canthal Tilt.

2. Volumetric Estimation: Calculates the surface area of cheek regions to detect potential muscle atrophy over time.

3. Live Feedback: Provides real-time visual bars showing left-vs-right symmetry balance.

4. Clinical-Style Reporting: Generates detailed text reports highlighting exactly which facial regions have drifted from the baseline.

5. Privacy : All images and biometric data are stored locally in CSV files on your machine. Nothing is uploaded to the cloud
## Technical Stack/Acknowledgements

Python 3.9+

MediaPipe: For dense 3D facial landmark detection.

OpenCV: For image processing and overlay rendering.

Tkinter: For the graphical user interface.

Pandas/NumPy: For data storage and vector calculus.
