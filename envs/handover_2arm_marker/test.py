import numpy as np
import mujoco
import os

os.environ.setdefault('MUJOCO_GL', 'glfw')

xml_path = os.path.join(os.path.dirname(__file__), 'scene.xml')
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


print('joint names:', [model.joint(i).name for i in range(model.njnt)])
print('total joints (njnt):', model.njnt)

# 'left/waist', 'left/shoulder', 'left/elbow', 'left/forearm_roll', 
# 'left/wrist_angle', 'left/wrist_rotate', 'left/left_finger', 'left/right_finger', 
# 'right/waist', 'right/shoulder', 'right/elbow', 'right/forearm_roll', 'right/wrist_angle', 
# 'right/wrist_rotate', 'right/left_finger', 'right/right_finger', 'middle_box_joint'

