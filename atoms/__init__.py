"""
A library for anomaly detection with multi-body robots.

  iNomaly applies an analytical approach to anomaly detection. It allows features manipulation to include information
  based on hardware, geometry, sensors and physics when training the anomaly detection classifier. For example:

    - data range and/or residuals between two different estimates;
    - relationships between different data;
    - consistency of data based on physics.

  The module trains a classifier including the above relationships, hence generalizing the result to a wider context
  than the task with whom is trained.

  Author: Gabriele Nava, gabriele.nava@iit.it, Dec. 2022
"""
from .linearMPC import LinearMPC
