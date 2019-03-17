#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import numpy as np
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 1
MODEL_INFERENCE = "model_inference/frozen_inference_graph.pb"

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []
        self.kd_tree = None
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(MODEL_INFERENCE)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.image_counter = 0
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        x_y_waypoints = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg.waypoints ]
        self.waypoints_2d = x_y_waypoints
        self.kd_tree = KDTree(x_y_waypoints)



    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        self.image_counter += 1
#        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
#        cv2.imwrite("test/img_%d.png" % self.image_counter, cv_image)
        #cv2.imwrite("./red_light/red_%d.jpg" % self.image_counter, cv_image)
        #rospy.logwarn(cv_image)
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        #print("Process traffic lights")
        #print (self.state)
        #print("\n\n")
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_wp_idx = -1
        if self.kd_tree:
            current_pose_2d = (pose.pose.position.x, pose.pose.position.y)
            closest_wp_idx = self.kd_tree.query(current_pose_2d)[1]
            closest_wp = self.waypoints_2d[closest_wp_idx]
            prev_wp = self.waypoints_2d[closest_wp_idx - 1]
            cl_prev_vect = np.array(closest_wp) - np.array(prev_wp)
            pose_cl_vect = np.array(current_pose_2d) - np.array(closest_wp)
            angle_sign = np.dot(cl_prev_vect, pose_cl_vect)
            if angle_sign > 0:
                # closest wp is behind
                closest_wp_idx = (closest_wp_idx + 1) % len(self.waypoints_2d)
                # get ready to publish waypoints

        return closest_wp_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # ONLY USE FOR SIMULATION NOW
#        return light.state
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose)

            #TODO find the closest visible traffic light (if one exists)
            lights_wp_idxs = []
            cl_light_idx = -1
            stop_line_idx = -1
            for l_idx, tl in enumerate(self.lights):
                line_pose = PoseStamped()
                line_pose.pose.position.x = stop_line_positions[l_idx][0]
                line_pose.pose.position.y = stop_line_positions[l_idx][1]
                cl_wp = self.get_closest_waypoint(line_pose)
                # check if the idx is good
                if cl_light_idx < 0 and cl_wp > car_position:
                    cl_light_idx = cl_wp
                    stop_line_idx = l_idx
                elif cl_wp > car_position and cl_wp < cl_light_idx:
                    cl_light_idx = cl_wp
                    stop_line_idx = l_idx
            # find the closest light idx
            if cl_light_idx >=  0:
                light_wp = cl_light_idx
                light = self.lights[stop_line_idx]

        if light:
            state = self.get_light_state(light)
            rospy.logwarn("LIGHT STATE %d " % state)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
