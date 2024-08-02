import cv2
import numpy as np
import torch

K = [[1766.131892, 0, 509.485626],
     [0, 1767.300947, 378.562232],
     [0, 0, 1]]
D = [-0.223694, 0.129762, 0.000719, -0.000284, 0, 0, 0, 0]


def px2xy(point, camera_k, camera_d,  z=1.0):
    MK = np.array(camera_k, dtype=float).reshape(3,3)
    MD = np.array(camera_d, dtype=float)
    point = np.array(point, dtype=float)
    pts_uv = cv2.undistortPoints(point, MK, MD) * z
    return pts_uv[0][0]


# take second element for sort
def takeFirst(elem):
    return elem[0]

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2corners(x):
    bottom_left = [x[0], x[1]]
    bottom_right = [x[2], x[1]]
    top_left = [x[0], x[3]]
    top_right = [x[2], x[3]]
    return bottom_left, bottom_right, top_left, top_right

def euler2quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]

def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0, 0, 255),2)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0, 0, 255),2)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0, 0, 255),2)
    return img

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))