#!/usr/bin/env python3
import tf
import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion

def costmap_callback(msg, tf_listener):

    width, height = msg.info.width, msg.info.height
    costmap_2d = np.array(msg.data).reshape((height, width))
    costmap_cv = ((costmap_2d + 1) / 101.0 * 255).astype(np.uint8)

    # Get the transformation
    (trans, rot) = tf_listener.lookupTransform('ego_vehicle', msg.header.frame_id, rospy.Time(0))

    # Convert quaternion to Euler
    roll, pitch, yaw = euler_from_quaternion(rot)

    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), np.degrees(-yaw), 1)
    rotated = cv2.warpAffine(costmap_cv, rotation_matrix, (width, height))

    # Apply second rotation
    rotated_costmap = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Crop out bottom half 
    cropped_costmap = rotated_costmap[:width//2, :]

    resized_costmap = cv2.resize(cropped_costmap, (1280, 720))

    '''print("Size of costmap_2d:", costmap_2d.shape)
    print("Size of costmap_cv:", costmap_cv.shape)'''    
    #print("Size of cropped costmap:", cropped_costmap.shape)
    #print("Size of resized_costmap:", resized_costmap.shape)

    return resized_costmap
    

    cv2.imshow("Cropped Local Costmap", resized_costmap)
    cv2.waitKey(1)

def main():
    rospy.init_node('costmap_listener', anonymous=True)
    tf_listener = tf.TransformListener()
    rospy.Subscriber('/planner/move_base/local_costmap/costmap', OccupancyGrid, costmap_callback, tf_listener)
    rospy.spin()

if __name__ == '__main__':
    main()


