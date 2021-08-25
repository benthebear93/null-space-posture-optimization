from scipy.spatial.transform import Rotation as R
# import numpy as np


# def euler2quat(Rx,Ry,Rz):
# 	r = R.from_euler('zyx', [90, 45, 30], degrees=True)
# 	r.as_quat()
# 	print(r)
# 	return r

# if __name__ =="__main__":
# 	euler2quat(30, 40, 45)


# Create a rotation object from Euler angles specifying axes of rotation
def euler2quat(Rx,Ry,Rz):
	rot = R.from_euler('zyx', [Rz, Ry, Rx], degrees=True)

	# Convert to quaternions and print
	rot_quat = rot.as_quat()
	return rot_quat

if __name__ =="__main__":
	print(euler2quat(30, 40, 45))
