#!/usr/bin/env python3
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
import math
import tf2_ros
import rospy
from std_msgs.msg import Float64,Header
import matplotlib.pyplot as plt
import cv2

wheelbase = 1.6

def convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase):
  if omega == 0 or v == 0:
    return 0

  radius = v / omega
  return math.atan(wheelbase / radius)

def callback_cmd(data):
   v = data.linear.x
   omega = data.angular.z
   steering_angle_rad = convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase)
   steering_angle_deg = math.degrees(steering_angle_rad)
   msg=Float64()
   msg.data = -steering_angle_deg
   steer_pub.publish(msg)


def main_loop():
    rospy.Subscriber('/planner/cmd_vel', Twist, callback_cmd)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('carla_obstacle_avoidance', anonymous=True)
    steer_pub = rospy.Publisher('KalmanAngle', Float64, queue_size=1)
    #map_pub=rospy.Publisher('myOccupancyMap',OccupancyGrid,queue_size=1)
    rate = rospy.Rate(10.0)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    print("------------------Hello carla obstacle avoidance------------------------------")

    while not rospy.is_shutdown():
        try:
            main_loop()
        except rospy.ROSException :
            rate.sleep()
            continue

