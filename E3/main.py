import numpy as np
import scipy.io
import cv2

data_2d = scipy.io.loadmat('Features2D.mat')
data_3d = scipy.io.loadmat('Features3D.mat')

points_2d = data_2d['f']
points_3d = data_3d['P'][:, :3]

points_2d = np.array(points_2d, dtype=np.float32)
points_3d = np.array(points_3d, dtype=np.float32)

camera_matrix = np.eye(3)
dist_coeffs = np.zeros((4, 1))

success, rotation_vector, translation_vector = cv2.solvePnP(
    points_3d, 
    points_2d, 
    camera_matrix, 
    dist_coeffs, 
    flags=cv2.SOLVEPNP_ITERATIVE
)

rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

projected_points, _ = cv2.projectPoints(
    points_3d, 
    rotation_vector, 
    translation_vector, 
    camera_matrix, 
    dist_coeffs
)

error = np.linalg.norm(points_2d - projected_points.squeeze(), axis=1)
mean_error = np.mean(error)
print(f"Mean reprojection error: {mean_error} pixels")

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
print(f"Estimated focal length: fx={fx}, fy={fy}")