import pybullet as p
import pybullet_data
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("tesbot.urdf", [0, 0, 0.3], useFixedBase=False)
