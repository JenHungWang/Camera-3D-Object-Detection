from math import frexp
from traceback import print_tb
from torch import imag
from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import math
import yaml
import numpy as np
from yolov5_ros2.global_settings import import_config_dict
from yolov5_ros2.cv_tool import px2xy, takeFirst, xywh2xyxy, xyxy2corners, draw_cube, rot2eul, euler2quaternion

package_share_directory = get_package_share_directory('yolov5_ros2')

# Load config
config_dict = import_config_dict()

# bbox coordinates
AXIS = np.reshape(np.float32(config_dict['COORDINATE']['axis']), (-1, 3))
CUBE = np.float32(config_dict['COORDINATE']['cube'])
RECT = np.float32(config_dict['COORDINATE']['rect'])

# 3D object points
WHEEL_REAR_LEFT = config_dict['COORDINATE']['wheel_rear_left']
WHEEL_REAR_RIGHT = config_dict['COORDINATE']['wheel_rear_right']
WHEEL_FRONT_LEFT = config_dict['COORDINATE']['wheel_front_left']
WHEEL_FRONT_RIGHT = config_dict['COORDINATE']['wheel_front_right']

class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        
        # load config
        model = package_share_directory + config_dict['PARAMETER']['model']
        device = config_dict['PARAMETER']['device']
        image_topic = config_dict['TOPIC']['image_topic']
        camera_info_file = package_share_directory + config_dict['PARAMETER']['camera_info_file']
        camera_info_topic = config_dict['TOPIC']['camera_info_topic']
        self.show_result = config_dict['PARAMETER']['show_result']
        yolo_detectoin_topic = config_dict['TOPIC']['yolo_detectoin_topic']

        # load model
        self.yolov5 = YOLOv5(model_path=model, device=device)

        pcl_qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                                              history=rclpy.qos.HistoryPolicy.KEEP_ALL,
                                              depth=1)

        # create publisher
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray, yolo_detectoin_topic, 10)
        self.result_msg = Detection2DArray()

        # create sub image
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, pcl_qos_policy)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)

        # get camera info
        with open(camera_info_file) as f:
            self.camera_info = yaml.full_load(f.read())
            print(self.camera_info['k'], self.camera_info['d'])

        # convert cv2 (cvbridge)
        self.bridge = CvBridge()

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info['k'] = msg.k
        self.camera_info['d'] = msg.d
        self.camera_info['r'] = msg.r
        self.camera_info['roi'] = msg.roi
        self.camera_info_sub.destroy()

    def image_callback(self, msg: Image):

        image = self.bridge.imgmsg_to_cv2(msg)
        # image = self.bridge.compressed_imgmsg_to_cv2(msg)
        # image_conv = cv2.cvtColor(image, cv2.COLOR_YUV420p2RGBA)
        # image_conv = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
        # image_conv = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_Y422)
        # image_conv = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYNV)
        # image_conv = cv2.cvtColor(image, cv2.COLOR_YUV2BGRA_UYVY)

        #TODO add argument for conf_thres and iou_thres
        detect_result = self.yolov5.predict(image, size=1280)
        self.get_logger().info(str(detect_result))

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()

        # parse results
        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # initial Parameters
        xywh_AV = []
        xywh_wheel = []
        img_point = []
        obj_point = []
        center = []
        yaw = []
        wheel_det = []
        ratio = []
        wheel_count = [0, 0]
        draw = True
        camera_matrix = np.reshape(np.float32(self.camera_info['k']), (3, 3))
        distortion_coefficient = np.float32(self.camera_info['d'])

        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            # print('test', name)
            detection2d = Detection2D()
            detection2d.id = name
            # detection2d.bbox
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0
            width = float(x2-x1)
            height = float(y2-y1)
            xywh_pixel = np.array([center_x, center_y, width, height], dtype=np.int16)
            detection2d.bbox.center.x = center_x
            detection2d.bbox.center.y = center_y
            detection2d.bbox.size_x = width
            detection2d.bbox.size_y = height

            # draw 2D bbox
            """
            if self.show_result:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, name, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            """

            if index == 0:
                xywh_AV = xywh_pixel
            else:
                xywh_wheel.append(xywh_pixel)

        if len(xywh_AV) != 0:
            for w in xywh_wheel:
                w_diff = w[0] - xywh_AV[0]
                h_diff = w[1] - xywh_AV[1]
                if abs(w_diff) < xywh_AV[2]/2 and abs(h_diff) < xywh_AV[3]/2:
                    wheel_det.append(w)
                    if w_diff <= 0:
                        wheel_count[0] += 1
                    else:
                        wheel_count[1] += 1
            wheel_det.sort(key=takeFirst)

            for x in range(len(wheel_det)):
                ratio_temp = wheel_det[x][3]/wheel_det[x][2]
                ratio.append(ratio_temp)

            if len(ratio) == 3:
                if ratio[1] < 1.35 and ratio[0] < ratio[2]:
                    wheel_count = [2, 1]

                if ratio[1] < 1.35 and ratio[0] > ratio[2]:
                    wheel_count = [1, 2]

            if wheel_count == [1, 1]:
                # build 2D-3D Correspondences
                points2d = xywh2xyxy(torch.tensor(wheel_det)).tolist()
                img_point.extend(xyxy2corners(points2d[0]))
                img_point.extend(xyxy2corners(points2d[1]))

                if ratio[0] < 1.3 and ratio[1] < 1.3:
                    obj_point.extend(WHEEL_FRONT_LEFT)
                    obj_point.extend(WHEEL_REAR_LEFT)
                    draw = False

                else:
                    obj_point.extend(WHEEL_REAR_LEFT)
                    obj_point.extend(WHEEL_REAR_RIGHT)
                    # improvement for BBox Projection
                    add_img_point = [img_point[0], img_point[2], img_point[5], img_point[7]]
                    img_point.extend(add_img_point)
                    add_obj_point = [[0, 3021, 612], WHEEL_FRONT_LEFT[2], [1930, 3021, 612], WHEEL_FRONT_RIGHT[3]]
                    obj_point.extend(add_obj_point)

                img_point = np.float32(img_point)
                obj_point = np.float32(obj_point)
            
            elif wheel_count == [2, 1]:
                # build 2D-3D Correspondences
                points2d = xywh2xyxy(torch.tensor(wheel_det)).tolist()
                img_point.extend(xyxy2corners(points2d[1]))
                img_point.extend(xyxy2corners(points2d[2]))
                img_point.extend(xyxy2corners(points2d[0]))
                img_point = np.float32(img_point)
                obj_point.extend(WHEEL_REAR_LEFT)
                obj_point.extend(WHEEL_REAR_RIGHT)
                obj_point.extend(WHEEL_FRONT_LEFT)
                obj_point = np.float32(obj_point)

            elif wheel_count == [1, 2]:
                # build 2D-3D Correspondences
                points2d = xywh2xyxy(torch.tensor(wheel_det)).tolist()
                img_point.extend(xyxy2corners(points2d[0]))
                img_point.extend(xyxy2corners(points2d[1]))
                img_point.extend(xyxy2corners(points2d[2]))
                img_point = np.float32(img_point)
                obj_point.extend(WHEEL_REAR_LEFT)
                obj_point.extend(WHEEL_REAR_RIGHT)
                obj_point.extend(WHEEL_FRONT_RIGHT)
                obj_point = np.float32(obj_point)

            if len(img_point) != 0:
                
                # PnP Pose Estimation
                # ret, rvecs, tvecs = cv2.solvePnP(obj_point, img_point, camera_matrix, distortion_coefficient,
                #                                  flags=cv2.SOLVEPNP_EPNP)
                ret, rvecs, tvecs = cv2.solvePnP(obj_point, img_point, camera_matrix, distortion_coefficient)

                # draw 3D BBox
                if draw and self.show_result:
                    # imgpts, jac = cv2.projectPoints(AXIS, rvecs, tvecs, CAMERA_MATRIX, DISTORTION)
                    imgpts, jac = cv2.projectPoints(CUBE, rvecs, tvecs, camera_matrix, distortion_coefficient)
                    imgpts = imgpts.astype(int)
                    # draw_axes(im0, origin, imgpts)
                    draw_cube(image, imgpts)

                # convert reference point to AV center
                if tvecs[2] < 0:
                    tvecs = [x / -1000.0 for x in tvecs]
                else:
                    tvecs = [x / 1000.0 for x in tvecs]
                rotmat, _ = cv2.Rodrigues(rvecs)
                transformation = np.vstack((np.hstack((rotmat, tvecs)), [0, 0, 0, 1]))
                center = np.dot(transformation, [[0.95], [2.44], [0.6], [1]])
                center = center[0:3]

                # yaw
                ang = rot2eul(rotmat)
                yaw = -ang[1]

                # quaternion
                quaternion = euler2quaternion(math.pi, 0, yaw)
                
                t_v = np.round(center, 4).tolist()
                obj_pose = ObjectHypothesisWithPose()
                obj_pose.hypothesis.class_id = "AV"
                obj_pose.hypothesis.score = float(scores[index])
                obj_pose.pose.pose.position.x = t_v[0][0]
                obj_pose.pose.pose.position.y = t_v[2][0]
                obj_pose.pose.pose.orientation.x = quaternion[3]
                obj_pose.pose.pose.orientation.y = quaternion[2]
                obj_pose.pose.pose.orientation.z = quaternion[1]
                obj_pose.pose.pose.orientation.w = quaternion[0]
                detection2d.results.append(obj_pose)
                self.result_msg.detections.append(detection2d)


        # if view or pub
        if self.show_result:
            cv2.imshow('result', image)
            cv2.waitKey(1)

        if len(categories) > 0:
            self.yolo_result_pub.publish(self.result_msg)


def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()


if __name__ == "__main__":
    main()