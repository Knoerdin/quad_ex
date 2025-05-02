import pybullet as p
import pybullet_data
import time
import random

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("box_bot.urdf", [0, 0, 0.3], useFixedBase=False)
p.setRealTimeSimulation(0)  # Turn OFF realtime for manual stepping

# Setup: create sliders once
joint_indices = []
param_ids = []

print("Joints:")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    joint_type = info[2]  # 0=revolute, 1=prismatic, 4=fixed
    name = info[1].decode()
    print(f"{i}: {name} ({'revolute' if joint_type==0 else 'fixed'})")

    if joint_type == p.JOINT_REVOLUTE:
        joint_indices.append(i)
        param_id = p.addUserDebugParameter(name, -1.57, 1.57, 0)
        param_ids.append(param_id)

# Simulation loop
while True:
    print_index = 0
    for i, joint_index in enumerate(joint_indices):
        target = p.readUserDebugParameter(param_ids[i])
        # target = random.uniform(-1.57, 1.57)
        p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, targetPosition=target, force=20)

    # get the height of the joints and the body
    joint_states = p.getJointStates(robot, joint_indices)
    joint_heights = [state[0] for state in joint_states]
    body_state = p.getBasePositionAndOrientation(robot)
    body_height = body_state[0][2]  # z-coordinate of the base position

    if max(joint_heights) > body_height:
        print("Warning: Joint height exceeds body height!")
        # reset the joint positions to be within the body height
        for i, joint_index in enumerate(joint_indices):
            p.setJointMotorControl2(robot, joint_index, p.POSITION_CONTROL, targetPosition=body_height, force=20)



    p.stepSimulation()
    time.sleep(1./240)
