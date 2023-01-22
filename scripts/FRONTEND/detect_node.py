#!/usr/bin/env python3
import rospy
import cv2
import sys, os
import numpy as np
from detector import detection_model, MetadataCatalog, Visualizer
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from measurement3d_2d import compute_line, compute_projected_point, compute_measurements
from tf.transformations import euler_from_matrix
 
# Importing Visualization UTILS
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'UTILS')) 
from visualize import visualize


class detection_node:

    def __init__(self, model_name, display_enabled = False, sim_enabled = False): # COMPLETED
        # ROS communication utilities
        rgb_topic = str(rospy.get_param("rgb_topic"))
        depth_topic = str(rospy.get_param("depth_topic"))
        measurement_topic = str(rospy.get_param("measurement_topic"))
        confidence = float(rospy.get_param("detection_confidence"))
        sampling_t = float(rospy.get_param("sampling_t"))
        self.maximum_depth = float(rospy.get_param("maximum_depth"))
        self.below_cam_th = float(rospy.get_param("below_cam_th"))
        self.inclination_th = np.deg2rad(float(rospy.get_param("inclination_th")))

        # Define machine learning model for detecting tree trunks
        self.DL_MODEL = detection_model(model_name, confidence)
        # Create a cv_bridge instance to convert from Image msg to CV img
        self.bridge = CvBridge()
        # Subscription to both RGB and DEPTH images
        self.rgb_sub = rospy.Subscriber(rgb_topic,CompressedImage, self.rgb_callback)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback) # real robot
        # Synchronous routine for computing measruments
        self.detect_routine = rospy.Timer(rospy.Duration(sampling_t), self.inference_callback) # Run Inference at 10 Hz
        # Publisher for resulting measurements
        self.measurement_pub = rospy.Publisher(measurement_topic,Float32MultiArray, queue_size = 1)
        self.measurement_msg = Float32MultiArray()

        # Publisher for Debuggin purposes
        self.rgb_detection_pub = rospy.Publisher("/rgb_keypoints",Image, queue_size = 1)
        self.depth_detection_pub = rospy.Publisher("/depth_keypoints",Image, queue_size = 1)
        self.visualizer = visualize()

        # Image Storage
        self.rgb_img = None
        self.depth_img = None
        self.rgb_img_shape = None
        self.depth_img_shape = None
        
        self.display_enabled = display_enabled
        self.sim_enabled = sim_enabled
        self.rgb_keypoints_list = []
        self.depth_keypoints_list = []

        # Intrinsic Camera Parameters
        self.fx_rgb = float(rospy.get_param("fx_rgb"))
        self.fy_rgb = float(rospy.get_param("fy_rgb"))
        self.cx_rgb = float(rospy.get_param("cx_rgb"))
        self.cy_rgb = float(rospy.get_param("cy_rgb"))

        self.fx_depth = float(rospy.get_param("fx_depth"))
        self.fy_depth = float(rospy.get_param("fy_depth"))
        self.cx_depth = float(rospy.get_param("cx_depth"))
        self.cy_depth = float(rospy.get_param("cy_depth"))

        self.rgb_M = np.array([[self.fx_rgb, 0.0, self.cx_rgb],
                               [0.0, self.fy_rgb, self.cy_rgb],
                               [0.0, 0.0, 1.0]])

        self.depth_M = np.array([[self.fx_depth, 0.0, self.cx_depth],
                                 [0.0, self.fy_depth, self.cy_depth],
                                 [0.0, 0.0, 1.0]])

        self.camera_wrt_base_link = np.array([0.4, 0, 0.1])

        self.rgb2depth = self.depth_M@np.linalg.pinv(self.rgb_M)


    def rgb_callback(self, data): # COMPLETED
        '''
        Callback used to store the most recent RGB image
        '''
        try:
            self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(data)
            if self.rgb_img_shape is None:
                self.rgb_img_shape = self.rgb_img.shape
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data): # COMPLETED
        '''
        Callback used to store the most recent depth image
        '''
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(data)
            if self.depth_img_shape is None:
                self.depth_img_shape = self.depth_img.shape
        except CvBridgeError as e:
            print(e)

    def inference_callback(self, event): # STILL DEVELOPMENT, REQUIRES CLEANING!!!!

        if self.rgb_img is None or self.depth_img is None:
            return None 

        # Grabbing the latest images received by the system
        rgb_img = self.rgb_img
        depth_img = self.depth_img

        # Execute inference with the DL model for deteting tree trunk keyoints
        outputs_pred = self.DL_MODEL.predict(rgb_img)
        # Get detected instaces (Three keypoints along the trunk)
        instances = outputs_pred['instances'].get('pred_keypoints').to("cpu").numpy()
        # Iterate over the instances 
        for instance in instances:
            # instance[:, 0] are the x along the width
            # instance[:, 1] are the y along the height
            # Getting coordinate indices from the rgb camera
            # for the keypoints
            ij = instance[[0, 3, 4], :]

            # Converting to depth img coordinates
            depth_ij = self.rgb2depth@np.vstack((ij[:, 0:2].T, np.ones(3)))
            depth_ij = np.abs(np.floor(depth_ij[0:2, :].T)).astype(np.uint32)

            if self.display_enabled:
                self.rgb_keypoints_list.append(ij.astype(np.uint32))
                self.depth_keypoints_list.append(depth_ij)

            # Set z as a column vector and convert it to meters
            z = depth_img[depth_ij[:, 1], depth_ij[:, 0]]

            if (z > 0.0).all() and (z < self.maximum_depth).all(): # if no measurements with zero values proceed to compute 3d points fro the depth measurements
                
                # Compute 3D point wrt to the camera frame given the depth measurements
                # Convert to meters
                Pc = (np.vstack(((ij[:, 0] - self.cx_rgb)/self.fx_rgb, (ij[:, 1] - self.cy_rgb)/self.fy_rgb, np.ones(3)))*z)/1000.0
                # Make the Points Homogeneous
                Pc = np.vstack((Pc, np.ones(3))) # [x, y, z, 1]'

                if (Pc[1, :] < self.below_cam_th).all(): # Discard points that are 0.3 meters below the camera frame (0.3 units in the positive y axis of the camera_depth_optical_frame)
                        
                    # Compute measurement given the detected instances
                    p0, d = compute_line(Pc.T[:, 0:3]) # Points in [x, y, z] format
                    projected_point_c = compute_projected_point(p0, d)
                    distance, angle, _, angle_y, _ = compute_measurements(projected_point_c, d)

                    # Only consider lines with less than 0.1 rad difference wrt the camera's y axis
                    if angle_y[0, 0] < self.inclination_th:

                        # Publish Measurment to SLAM node
                        self.measurement_msg.data = np.asarray([distance, angle], dtype=np.float32)
                        self.measurement_pub.publish(self.measurement_msg)

                        if self.sim_enabled:
                            self.point3Dvis(Pc)                 # Display Detected 3D points (for simulation only)
                            self.cylinder3Dvis(distance, angle) # Display detected landmark according to the range and angle measured

        if self.display_enabled:
            # Show estimated tree trunk keypoints
            self.show_keypoints(rgb_img, depth_img)
            

    def point3Dvis(self, Pc):
        # Ground Truth Transform for Visualization and debugging (MEasured 3D points)
        T = self.visualizer.get_transform('camera_depth_optical_frame', 'odom')
        Pw = T@Pc
        self.visualizer.draw_point(Pw[:, [0]])
        self.visualizer.draw_point(Pw[:, [1]])
        self.visualizer.draw_point(Pw[:, [2]])

    def cylinder3Dvis(self, distance, angle):
        # Ground Truth Transform for Visualization and debugging (Measured 3D landmark)
        T = self.visualizer.get_transform('base_link', 'odom')
        _, _ , yaw = euler_from_matrix(T[0:3, 0:3])
        lx = T[0, 3] + (distance + self.camera_wrt_base_link[0])*np.cos(angle + yaw)
        ly = T[1, 3] + (distance + self.camera_wrt_base_link[0])*np.sin(angle + yaw) 
        self.visualizer.draw_cylinder(np.asarray([[lx],[ly]]))

    def show_keypoints(self, rgb_img, depth_img) :

        for i in range(len(self.rgb_keypoints_list)):

            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][0, 0], self.rgb_keypoints_list[i][0, 1]), 5, (255, 0, 0), 2)
            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][1, 0], self.rgb_keypoints_list[i][1, 1]), 5, (255, 0, 0), 2)
            cv2.circle(rgb_img, (self.rgb_keypoints_list[i][2, 0], self.rgb_keypoints_list[i][2, 1]), 5, (255, 0, 0), 2)

            cv2.circle(depth_img, (self.depth_keypoints_list[i][0, 0], self.depth_keypoints_list[i][0, 1]), 5, (20000, 20000, 20000), 2)
            cv2.circle(depth_img, (self.depth_keypoints_list[i][1, 0], self.depth_keypoints_list[i][1, 1]), 5, (20000, 20000, 20000), 2)
            cv2.circle(depth_img, (self.depth_keypoints_list[i][2, 0], self.depth_keypoints_list[i][2, 1]), 5, (20000, 20000, 20000), 2)

        self.rgb_detection_pub.publish(self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8"))
        self.depth_detection_pub.publish(self.bridge.cv2_to_imgmsg(depth_img, encoding="mono16"))
        self.rgb_keypoints_list = []; self.depth_keypoints_list = []


def main(args): # COMPLETED
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir,"MODELS/R-50_RGB_60k.pth")
    rospy.init_node('DETECT_NODE', anonymous=True)
    display_enabled = 'True' == args[1]
    sim_enabled = 'True' == args[2]
    ic = detection_node(model_name, display_enabled, sim_enabled)
    print("Model", model_name, "Loaded")
    print("NODE STARTED -- 'detect_node.py'")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
