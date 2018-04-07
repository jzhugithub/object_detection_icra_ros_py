#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from detect_image import DetectImage

from object_detection_icra_ros_py.msg import RobotImagePos

class DetectVideo(object):
    # parameters need to modify
    # node
    subscribed_topic = '/my_video'
    # video_output
    show_video_flag = True
    save_video_flag = False
    video_rate = 30.0
    video_output_path = os.path.join(os.path.abspath('../video_output'), 'out.avi')
    VIDEO_WINDOW_NAME = 'video_output'
    # detect
    # Create DetectImage class
    OBJECT_DETECTION_PATH = '/home/zj/program/models/object_detection'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = '/home/zj/ros_workspace/src/object_detection_icra_ros_py/model/pack.pb'
    # PATH_TO_CKPT = '/home/zj/database/fisheye2_data/model/ssd05re0208/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/home/zj/ros_workspace/src/object_detection_icra_ros_py/model/pack.pbtxt'
    # PATH_TO_LABELS = '/home/zj/database_temp/fisheye2_data_set/fisheye2_label_map.pbtxt'
    NUM_CLASSES = 3
    # NUM_CLASSES = 2


    # parameters do not need to modify
    # node
    # image_sub_ = rospy.Subscriber()
    # video_output
    image_hight = -1
    image_width = -1
    video = 'VideoWriter'
    # frame
    frame_num = 1
    src = np.array([])
    dst = np.array([])
    cvi = CvBridge()
    # detect
    di = 'DetectImage'

    def __init__(self):
        # node
        self.image_sub_ = rospy.Subscriber(self.subscribed_topic, Image, self.image_callback, queue_size=1)
        # video_output
        if self.show_video_flag:
            cv2.namedWindow(self.VIDEO_WINDOW_NAME)
        # detect
        self.di = DetectImage(self.PATH_TO_CKPT, self.PATH_TO_LABELS, self.NUM_CLASSES, self.show_video_flag)
        # pub
        self.pub = rospy.Publisher('/robot_image_pos', RobotImagePos, queue_size=10)

    def __del__(self):
        cv2.destroyAllWindows()

    def image_callback(self, msg):
        try:
            self.src = self.cvi.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return
        if self.frame_num == 1:
            self.image_hight, self.image_width, channels = self.src.shape
            self.video = cv2.VideoWriter(self.video_output_path, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'),
                                         int(self.video_rate), (int(self.image_width), int(self.image_hight)))
        self.frame_num += 1

        # detect
        # self.src_3 = cv2.resize(self.src_3,(160, 120))
        cv2.cvtColor(self.src, cv2.cv.CV_BGR2RGB, self.src)  # since opencv use bgr, but tensorflow use rbg
        self.dst, detectmsg = self.di.run_detect(self.src)
        cv2.cvtColor(self.dst, cv2.cv.CV_RGB2BGR, self.dst)  # since opencv use bgr, but tensorflow use rbg

        # pub
        pubmsg = RobotImagePos()
        pubmsg.exist_enemy_flag = detectmsg['exist_enemy_flag']
        pubmsg.exist_pad_flag = detectmsg['exist_pad_flag']
        pubmsg.exist_friend_flag = detectmsg['exist_friend_flag']
        pubmsg.enemy_num = detectmsg['enemy_num']
        pubmsg.pad_num = detectmsg['pad_num']
        pubmsg.friend_num = detectmsg['friend_num']
        # ymin, xmin, ymax, xmax
        pubmsg.enemy_ymin, pubmsg.enemy_xmin, pubmsg.enemy_ymax, pubmsg.enemy_xmax = detectmsg['enemy_msg']
        pubmsg.pad_ymin, pubmsg.pad_xmin, pubmsg.pad_ymax, pubmsg.pad_xmax = detectmsg['pad_msg']
        pubmsg.friend_ymin, pubmsg.friend_xmin, pubmsg.friend_ymax, pubmsg.friend_xmax = detectmsg['friend_msg']


        self.pub.publish(pubmsg)

        # save and show video_output
        if self.save_video_flag:
            self.video.write(self.dst)
        if self.show_video_flag:
            cv2.imshow(self.VIDEO_WINDOW_NAME, self.dst)
            cv2.waitKey(1)


if __name__ == '__main__':
    print('opencv: ' + cv2.__version__)
    rospy.init_node('object_detection_icra_ros_py_node', anonymous=True)
    mrd = DetectVideo()
    rospy.spin()
