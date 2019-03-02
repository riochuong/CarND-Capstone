#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.dbw_enabled = False
        self.current_twist = None
        self.command_twist = None
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self._dbw_status_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self._twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self._velocity_cb)
        # TODO: Create `Controller` object
        self.controller = Controller(
            wheel_radius=wheel_radius,
            wheel_base= wheel_base,
            steer_ratio=steer_ratio,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle,
            min_speed=0.,
            deccel_limit=decel_limit,
            accel_limit=accel_limit,
            vehicle_mass=vehicle_mass
        )

        self.loop()

    def _dbw_status_cb(self, status):
        self.dbw_enabled = status

    def _twist_cb(self, msg):
        self.command_twist = msg

    def _velocity_cb(self, msg):
        self.current_twist = msg

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            if self.current_twist is not None and self.command_twist is not None:
                current_vx = self.current_twist.twist.linear.x
                current_vy = self.current_twist.twist.linear.y
                current_ang_vel = self.current_twist.twist.angular.z
                desired_vx = self.command_twist.twist.linear.x
                desired_vy = self.command_twist.twist.linear.y
                desired_ang_vel = self.command_twist.twist.angular.z

                throttle, brake, steer = self.controller.control(
                    current_vx=current_vx,
                    current_vy=current_vy,
                    current_ang_vel=current_ang_vel,
                    desired_vx=desired_vx,
                    desired_vy=desired_vy,
                    desired_ang_vel=desired_ang_vel
                )
                if self.dbw_enabled:
                    self.publish(throttle, brake, steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
