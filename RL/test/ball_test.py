import pybullet as p
import numpy as np
import pybullet_data
import utils_tx90
import math as m
from time import sleep

BallPath="C:/Users/nswve/Desktop/git/null-space-posture-optimization/urdf/ball.urdf"
print("==========loading URDF==========")
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

robotStartPos = [0,0,0.0]
robotStartOrn = p.getQuaternionFromEuler([0,0,0])

robotID = p.loadURDF(BallPath, robotStartPos, robotStartOrn, useFixedBase = True, flags=p.URDF_USE_INERTIA_FROM_FILE)
sleep(100000)