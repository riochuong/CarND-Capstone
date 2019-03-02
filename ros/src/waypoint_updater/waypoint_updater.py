#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import numpy as np
import math
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        self.base_waypoints = None
        self.kd_tree = None
        self.current_pose = None
        rospy.init_node('waypoint_updater')
        rospy.loginfo("======= Initialize Waypoint Updater !!!!!!!!!!!!!1")
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.publish_update_waypoints()

    def publish_update_waypoints(self):
        rate = rospy.Rate(50)
        print("Publish new waypoints")
        while not rospy.is_shutdown():
            if (self.base_waypoints is not None) and (self.current_pose is not None):
                # find the nearest neighborhood that after pose
                current_pose_2d = (self.current_pose.pose.position.x, self.current_pose.pose.position.y)
                print("Publish new waypoints")
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
                publish_waypoints = self.base_waypoints.waypoints[closest_wp_idx:closest_wp_idx + LOOKAHEAD_WPS]
                lane = Lane()
                lane.header = self.base_waypoints.header
                lane.waypoints = publish_waypoints
                self.final_waypoints_pub.publish(lane)
            rate.sleep()



    def pose_cb(self, msg):
        self.current_pose = msg

    def waypoints_cb(self, msg):
        print("Received Base waypoints !")
        self.base_waypoints = msg
        x_y_waypoints = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg.waypoints ]
        self.waypoints_2d = x_y_waypoints
        self.kd_tree = KDTree(x_y_waypoints)




    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
