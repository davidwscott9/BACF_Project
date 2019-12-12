# Learning Background-Aware Correlation Filters for Visual Tracking: An Implementation in Python
This project is based on the ICCV 2017 Paper by Galoogahi et al. entitled Learning Background-Aware Correlation Filters for Visual Tracking.
This project was completed for EECS 5323 at York University in 2019. 

## Motivation
The goal of this project is to convert the work described in the original paper to Python, an open-source programming language. The results outlined in the original paper were created using MATLAB which is not an open-source language. The lack of open-source availability makes this work more challenging to iterate and improve upon as many do not have access to the proprietary MATLAB software. As such, one would need to implement the code from the basic mathematical formulae outlined in the paper in order to achieve the baseline results. Thus, it is expected that converting the code to Python will help make this work more accessible to many, allowing for future collaboration and innovation on this work.

## Requirements
### Downloading the benchmark datasets
First, the benchmark datasets need to be downloaded from [here](https://drive.google.com/file/d/1D2Vl9LQ6D2ROga7hR3RX7UdtOKFq1v00/view?usp=sharing). This package includes the OTB50, OTB100, and TC128 datasets.
Unzip the package and add the folder 'seq' to your Python Path.

### Python Version
This code was created and tested on Python 3.7

## Running the Project
### Single Sequence Demo
Run BACF_Demo_single_seq.py

### Benchmark Tests
Run benchmark_tests_script.py

## Required External Libraries
numpy
matplotlib
opencv-python
