#!/usr/bin/env python3
import rospy
import carla
from std_msgs.msg import Float64
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
import time
import math

current_speed = 0.0
max_steering_angle = 22.5  

"""def throttle_control(current_speed, target_speed, kp=.3):
    speed_error = target_speed - current_speed
    throttle = kp * speed_error
    return throttle"""

def callback_vehicle_status(data):
    global current_speed
    current_speed = data.velocity

def callback_steering_angle(data):
    global current_speed
    target_speed = 10 / 3.6
    steering_angle = data.data
    normalized_steering_angle = steering_angle / max_steering_angle
    control = CarlaEgoVehicleControl()
    #control.throttle = throttle_control(current_speed, target_speed)
    control.throttle = 0.15
    control.brake = 0
    control.steer = normalized_steering_angle
    control_publisher.publish(control)

    print(f"{steering_angle = }")
    print(f"{control.throttle = }")  
    print(f"{control.steer = }") 
    print(f"{current_speed = }")

    #time.sleep(0.25)

if __name__ == '__main__':
    rospy.init_node('carla_steering_brake', anonymous=True)
    control_publisher = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd', CarlaEgoVehicleControl, queue_size=1)
    rospy.Subscriber('KalmanAngle', Float64, callback_steering_angle)
    rospy.Subscriber('/carla/ego_vehicle/vehicle_status', CarlaEgoVehicleStatus, callback_vehicle_status)

    print("------------------Hello carla steering brake------------------------------")

    rospy.spin()

