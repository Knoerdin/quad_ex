import pybullet as p
import pybullet_data
import time

# run box_bot.urdf in pybullet
# create a connection to the physics server
p.connect(p.GUI,options="--width=1280 --height=720 --msaa=4")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# load the plane
plane_id = p.loadURDF("plane.urdf")

# load the plane and the robot
robot = p.loadURDF("connected_bot.urdf", [0, 0, 0.3], useFixedBase=False)
# set the time step
p.setTimeStep(0.01)
# set the simulation to run in real time
p.setRealTimeSimulation(1)

print("Simulation running. Press Ctrl+C to exit.")

try:
    while True:
        # Set the base's linear velocity to move forward along the x-axis.
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    print("Simulation terminated.")
    p.disconnect()