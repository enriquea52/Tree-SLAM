#! /usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
import tf
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_about_axis

class visualize:

    def __init__(self):

        self.frame_id = "odom"

        self.points_topic = str(rospy.get_param("points_topic"))
        self.cylinder_topic = str(rospy.get_param("cylinder_topic"))
        self.robot_vis_topic = str(rospy.get_param("robot_vis_topic"))
        self.landmark_vis_topic = str(rospy.get_param("landmark_vis_topic"))
        self.points_pub = rospy.Publisher(self.points_topic, Marker, queue_size = 2)
        self.cylinder_pub = rospy.Publisher(self.cylinder_topic, Marker, queue_size = 2)
        self.robot_cov = rospy.Publisher(self.robot_vis_topic, Marker, queue_size = 2)
        self.marker_array_pub = rospy.Publisher(self.landmark_vis_topic, MarkerArray, queue_size = 2)

        self.id_cyl = 0
        self.id_pt = 0
        self.listener = tf.TransformListener()
        self.origin, self.xaxis, self.yaxis, self.zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

        self.marker_array = MarkerArray()
        
    def draw_cylinder(self, point):

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 3
        marker.id = self.id_cyl

        # Set the scale of the marker
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 5

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point[0, 0]
        marker.pose.position.y = point[1, 0]
        marker.pose.position.z = 0

        print("publishing landmark")

        self.cylinder_pub.publish(marker)
        self.id_cyl += 1

    def draw_point(self, point):

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = self.id_pt

        # Set the scale of the marker
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point[0, 0]
        marker.pose.position.y = point[1, 0]
        marker.pose.position.z = point[2, 0]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.points_pub.publish(marker)
        self.id_pt += 1

    def draw_ellipse(self, pose, cov, id = 0, p = 0.95, color = (0.0, 0.0, 1.0, 1.0), publish = True):

        s = -2 * np.log(1 - p)

        lambda_, v = np.linalg.eig(s*cov)

        lambda_ = np.sqrt(lambda_)

        angle = np.arccos(v[0, 0])

        qz = quaternion_about_axis(angle, self.zaxis)

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = id

        # Set the scale of the marker
        marker.scale.x = lambda_[0]
        marker.scale.y = lambda_[1]
        marker.scale.z = 0.001

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.5

        # Set the pose of the marker
        marker.pose.position.x = pose[0, 0]
        marker.pose.position.y = pose[1, 0]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = qz[0]
        marker.pose.orientation.y = qz[1]
        marker.pose.orientation.z = qz[2]
        marker.pose.orientation.w = qz[3]

        if publish:
            self.robot_cov.publish(marker)

        else:
            return marker

    def draw_ellipses(self):
        self.marker_array_pub.publish(self.marker_array)
        self.marker_array.markers = []

    def get_transform(self, from_, to_):
        (trans,rot) = self.listener.lookupTransform(to_, from_, rospy.Time(0))
        T =  quaternion_matrix(rot)
        T[0, 3] = trans[0]
        T[1, 3] = trans[1]
        T[2, 3] = trans[2]
        return T

