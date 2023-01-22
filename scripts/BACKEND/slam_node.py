#!/usr/bin/env python3
import rospy
import sys, os
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from EKF import *

 # Importing Visualization UTILS
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, 'UTILS')) 
from visualize import visualize

# Node example class.
class SLAM_NODE():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self, odometry_topic, measurement_topic, M_DIST_TH, STATE_SIZE, LM_SIZE, LM_NUM, Q, R):
        # SLAM definitions
        self.SLAM = EKF(STATE_SIZE, LM_SIZE, LM_NUM, M_DIST_TH)
        # ROS utilities declarations
        self.predict_sub = rospy.Subscriber(odometry_topic, Odometry, self.predict)
        self.update_sub = rospy.Subscriber(measurement_topic,Float32MultiArray, self.update)
        # Setting covariance matrices to be used on SLAM
        self.SLAM.setQ(Q)
        self.SLAM.setR(R)
        # Previous time initialization to compute DT (change in time between odometry msgs)
        self.prev_time = None
        # Visualization class
        self.visualize = visualize()

    def predict(self, msg):

        if self.prev_time is None:
            self.prev_time = msg.header.stamp
            return None

        # GET DT (change in time)
        current_time = msg.header.stamp
        DT = current_time - self.prev_time
        self.prev_time = current_time

        # GET INPUT u AND COMPUTE THE PREDITION STEP
        ud = np.asarray([[msg.twist.twist.linear.x],
                         [msg.twist.twist.angular.z]])

        self.SLAM.prediction(ud, DT.to_sec())

        # CODE SECTION TO VISUALIZE ROBOT'S POSITION AND COVARIANCE
        robot_state = self.SLAM.Xr()
        robot_cov = self.SLAM.Prr()[0:2, 0:2]
        self.visualize.draw_ellipse(robot_state, robot_cov)

    def update(self, msg):
        
        if self.SLAM.NUM_LM < 1 :   # ADD INITIAL LANDMARK
            self.SLAM.new_landmark(msg.data)

        else:
            # EXECUTE DATA ASSOCIATION VIA MAHALANOBIES DISTANCE
            ID, MIN_DIS = self.SLAM.data_association(msg.data)

            # UPDATE STEP
            if MIN_DIS < self.SLAM.M_DIST_TH:
                # print("Updating", ID)
                self.SLAM.update(msg.data, ID)
            # ADD NEW LANDMARK
            else:
                # print("Adding new landmark")
                self.SLAM.new_landmark(msg.data)

        # CODE SECTION TO VISUALIZE LANDMARK POSITIONS AND COVARIANCES
        for idx in range(self.SLAM.NUM_LM):
            lm_state = self.SLAM.xl(idx).reshape(2, 1)
            lm_cov = self.SLAM.Pll(idx)
            marker_ = self.visualize.draw_ellipse(lm_state, lm_cov, id = idx+1, color = (1.0, 0.0, 0.0, 1.0), publish = False)
            self.visualize.marker_array.markers.append(marker_)
        self.visualize.draw_ellipses()


def main(args):
    # Initialize the node and name it.
    rospy.init_node('slam_node')
    # Go to class functions that do all the heavy lifting. Do error checking.
    M_DIST_TH =  float(rospy.get_param("mahalanobis_threshold")) # Threshold of Mahalanobis distance for data association. 10 is the best
    STATE_SIZE = int(rospy.get_param("robot_state_size"))        # State size [x,y,yaw]
    LM_SIZE =    int(rospy.get_param("landmark_state_size"))     # LM state size [x,y]
    LM_NUM =     int(rospy.get_param("max_landmarks"))           # Number of landmarks
    sigma_v =    float(rospy.get_param("motion_sigma_v"))
    sigma_w =    float(rospy.get_param("motion_sigma_w"))
    sigma_r =    float(rospy.get_param("measurement_sigma_r"))
    sigma_a =    float(rospy.get_param("measurement_sigma_a"))
    odometry_topic = str(rospy.get_param("odometry_topic"))
    measurement_topic = str(rospy.get_param("measurement_topic"))

    Q = np.diag([sigma_v, np.deg2rad(sigma_w)])**2 # Motion Uncertainty
    R = np.diag([sigma_r, np.deg2rad(sigma_a)])**2 # Measurement Uncertainty

    print("Mdometry Topic:", odometry_topic)
    print("Measurement Topic:", measurement_topic)
    print("M_DIST_TH:", M_DIST_TH)
    print("STATE_SIZE:", STATE_SIZE)
    print("LM_SIZE:", LM_SIZE)
    print("LM_NUM:", LM_NUM)
    print("Q", Q)
    print("R", R)

    node = SLAM_NODE(odometry_topic, measurement_topic, M_DIST_TH, STATE_SIZE, LM_SIZE, LM_NUM, Q, R)
    print("NODE STARTED -- 'slam_node.py'")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

# Main function.
if __name__ == '__main__':
    main(sys.argv)
    
