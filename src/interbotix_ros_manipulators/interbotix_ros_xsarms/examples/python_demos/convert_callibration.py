import numpy as np

# The manual math to replace Scipy
def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

# Create an empty 4x4 Identity Matrix
T_WC_head = np.eye(4)

# 1. Your exact XYZ translation
T_WC_head[0:3, 3] = [-0.007199387157014589, 0.03599075115461841, 0.05723884042607784]

# 2. Your exact Quaternion rotation
q = [-0.4896024815110257, 0.48077887946116327, -0.507338116649699, 0.5212956114880025]
T_WC_head[0:3, 0:3] = quat_to_matrix(q)

# Save the final file!
np.save('T_WC_head.npy', T_WC_head)

print("Calibration successfully converted! Here is your matrix:")
print(T_WC_head)