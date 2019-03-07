from yaw_controller import YawController
from pid import PID
from math import sqrt
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
import rospy
import numpy as np
from lowpass import LowPassFilter


class Controller(object):
    def __init__(self, **kwargs):
        wheel_base = kwargs.get('wheel_base', None)
        assert wheel_base is not None
        steer_ratio = kwargs.get('steer_ratio', None)
        assert steer_ratio is not None
        max_lat_accel = kwargs.get('max_lat_accel', None)
        assert max_lat_accel is not None
        max_steer_angle = kwargs.get('max_steer_angle', None)
        assert max_steer_angle is not None
        min_speed = kwargs.get('min_speed', 0)
        self.deccel_limit = kwargs.get('deccel_limit', None)
        self.accel_limit = kwargs.get('accel_limit', None)
        self.vehicle_mass = kwargs.get('vehicle_mass', None)
        self.wheel_radius = kwargs.get('wheel_radius', None)
        assert self.vehicle_mass is not None
        assert self.deccel_limit is not None
        assert self.accel_limit is not None
        # yaw will be control by yaw controller
        self.steering_controller = YawController(wheel_base, steer_ratio,
                                            min_speed, max_lat_accel, max_steer_angle)

        # throlttle will be controll by PID
        self.throttle_controller = PID(kp=0.6,
                                       ki=0.05,
                                       kd=0.1,
                                       mn=0,
                                       mx=1)
        self._vel_filter = LowPassFilter(tau=0.5 , ts=0.02)
        self._current_time = rospy.get_time()


    def control(self, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        current_vx = kwargs.get('current_vx', None)
        current_vy = kwargs.get('current_vy', None)
        current_ang_vel = kwargs.get('current_ang_vel', None)
        desired_vx = kwargs.get('desired_vx', None)
        desired_vy = kwargs.get('desired_vy', None)
        desired_ang_vel = kwargs.get('desired_ang_vel', None)

        # get delta time
        current_time = rospy.get_time()
        delta_time = current_time - self._current_time
        self._current_time = current_time

        # get new throttle value
        desired_linear_vel = np.sqrt(desired_vx**2 + desired_vy**2)
        current_linear_vel = np.sqrt(current_vx**2 + current_vy**2)
        # filter current velocity
        rospy.logwarn("Desire Vel %.2f" % desired_linear_vel)
        #current_linear_vel = self._vel_filter.filt(current_linear_vel)
        linear_vel_error = desired_linear_vel - current_linear_vel
        throttle = self.throttle_controller.step(linear_vel_error, delta_time)
        steering_ang = self.steering_controller.get_steering(desired_linear_vel,
                                                        desired_ang_vel, current_linear_vel)
        brake = 0
        # stop at red light
        if desired_linear_vel == 0. and current_linear_vel < 0.05:
            rospy.logwarn("Apply Break")
            throttle = 0
            brake = 400.
        elif throttle < 0.1 and linear_vel_error < 0:
            # we need to apply brake here
            deccel_rate = max(self.deccel_limit, linear_vel_error)
            brake = np.abs(deccel_rate) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering_ang
