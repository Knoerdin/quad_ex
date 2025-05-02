from mujoco import mjcf

mjcf_model = mjcf.from_urdf("tesbot.urdf")
for joint in mjcf_model.find_all("joint"):
    if joint.joint_type in ("hinge", "slide"):
        mjcf_model.actuator.add(
            "motor",
            name=f"motor_{joint.name}",
            joint=joint.name,
            gear=1.0
        )
mjcf_model.save("quad_ex.xml")