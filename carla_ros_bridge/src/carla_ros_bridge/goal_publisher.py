#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal():
    rospy.init_node('goal_publisher', anonymous=True)
    
    #rospy.sleep(10)
    # Create a publisher for the /move_base_simple/goal topic
    goal_publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    
    # Create a PoseStamped message with the desired goal information
    goal_msg = PoseStamped()
    goal_msg.header.stamp = rospy.Time.now()
    goal_msg.header.frame_id = "ego_vehicle"
    goal_msg.pose.position.x = 8.3
    goal_msg.pose.position.y = -2.27
    goal_msg.pose.position.z = 0.0
    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.010746232144959926
    goal_msg.pose.orientation.w = 0.9999422575802498
    
    # Publish the goal message
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        goal_publisher.publish(goal_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_goal()
    except rospy.ROSInterruptException:
        pass
