import pybullet as p
import pybullet_data
import time

# create a connection to the physics server
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# load the plane and the robot
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("main body.urdf", [0, 0, 0.3], useFixedBase=False)

p.setRealTimeSimulation(0)  # Turn OFF realtime for manual stepping

