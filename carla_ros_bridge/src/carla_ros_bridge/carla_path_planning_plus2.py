#!/usr/bin/env python3
import sys
import os
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion

sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.dirname(__file__))

import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseActionGoal
import cv2
from cv_bridge import CvBridge
import time
import torch

import math
import numpy as np
from torchvision import transforms
from HybridNets.backbone import HybridNetsBackbone
from HybridNets.utils.utils import letterbox, Params
import matplotlib.pyplot as plt
from datetime import datetime

use_cuda = torch.cuda.is_available()
#-----------------change on each computer-------------------------------------------------------------------------------------
def find_weights(filename):
    for root,dirs,files in os.walk("/home"):
        if filename in files:
            return os.path.join(root,filename)
    return None

path= find_weights("bdd100k.yml")
print(path)
if path is not None:
    params = Params(path)
else:
    raise Exception()
#-----------------------------------------------------------------------------------------------------------------------------
color_list_seg = {}
steer_msg = Float64()

for seg_class in params.seg_list:
    color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
shapes = [((720, 1280), ((0.5, 0.5), (0.0, 12.0)))]

DEBUG = False
SIMULATION = False
LANE_METERS = 9.8
Y_METERS = {
            10.0 : 500,
            7.5 : 565
            }
LANE_PIXELS = None
LATERAL_DISTANCE = 0
scale_factor = None
black_regions = None
y_black = None
prev_curvature = None
costmap = None
goal_published = False

wheelbase = 1.6  # Wheelbase of the vehicle

print(cv2.getBuildInformation())

# utils_path_v8
def load_camera_calib(sim=True):
    if not sim:
        # for the D435i camera
        mtx = [[914.05810546875, 0.0, 647.0606689453125],
            [0.0, 912.9447021484375, 364.1457824707031],
            [0.0, 0.0, 1.0 ]]
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # for the simulation
        mtx = [[1395.35, 0, 640],
                [0, 1395.35, 360],
                [0, 0, 1]]
        dist = [0, 0, 0, 0, 0]
    return np.array(mtx), np.array(dist)

def undistort(img, mtx, dist):
    '''
    Undistorts an image
    :param img (ndarray): Image, represented an a numpy array
    :param mtx: Camera calibration matrix
    :param dist: Distortion coeff's
    :return : Undistorted image
    '''
    
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    return undistort

def warp_image(img, warp_shape, src, dst):
    '''
    Performs perspective transformation (PT)
    :param img (ndarray): Image
    :param warp_shape: Shape of the warped image
    :param src (ndarray): Source points
    :param dst (ndarray): Destination points
    :return : Tuple (Transformed image, PT matrix, PT inverse matrix)
    '''
    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(img, M, warp_shape, flags=cv2.INTER_CUBIC)
    return warped, M, invM

def eye_bird_view(img, mtx, dist, d=530):
    ysize = img.shape[0]
    xsize = img.shape[1]
    
    undist = undistort(img, mtx, dist)
    src = np.float32([                   # Simulation (1280, 720)
        (694.0, 350.0),
        (586.0, 350.0),
        (50.0, 675.0),
        (1230.0, 675.0)
    ])

    dst = np.float32([
        (xsize - d, 0),
        (d, 0),
        (d, ysize),
        (xsize - d, ysize)
    ])

    warped, M, invM = warp_image(undist, (xsize, ysize), src, dst)
    return warped


def processing_mask(mask, img, show=False, d=530):
    global black_regions, y_black
    mtx, dist = load_camera_calib(sim=SIMULATION)
    warped = eye_bird_view(mask, mtx, dist, d=d)

    if black_regions is None:
        img_warped = eye_bird_view(img, mtx, dist, d=d)
        black_regions = cv2.inRange(img_warped, np.array([0, 0, 0]), np.array([0, 0, 0]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        black_regions = cv2.dilate(black_regions, kernel, iterations=1)
        y_black = np.min(np.nonzero(black_regions[:, 0] == 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14))
    res_morph = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)
    
    _, res_morph_th = cv2.threshold(res_morph, 0, 255, cv2.THRESH_BINARY)
    line_edges = cv2.Canny(res_morph_th, 100, 100)
    
    vertical_edges = np.zeros_like(line_edges)
    vertical_edges[:, [0, -1]] = 255

    combined_edges = cv2.bitwise_and(warped, vertical_edges)
    _, combined_edges = cv2.threshold(combined_edges, 0, 255, cv2.THRESH_BINARY)
    line_edges -= black_regions
    line_edges = cv2.bitwise_or(line_edges, combined_edges)
    if any(combined_edges[[y_black, y_black-10], 0] == 255):
        line_edges[:, 0] = 255
    elif any(combined_edges[[y_black, y_black-10], -1] == 255):
        line_edges[:, -1] = 255
    _, line_edges = cv2.threshold(line_edges, 2, 255, cv2.THRESH_BINARY)

    if show:
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(combined_edges)
        plt.imshow(line_edges)
        plt.show()
    return line_edges

def merge_close_edges(lst, tol=10):
    result = []
    temp = []
    for i, x in enumerate(lst):
        if temp and abs(x - temp[0]) > tol:
            if len(temp) == 1:
                result.append(temp[0])
            else:
                result.append(sum(temp) // len(temp))
            temp = []
        temp.append(x)
    if len(temp) == 1:
        result.append(temp[0])
    elif temp:
        result.append(sum(temp) // len(temp))
    return result

def computing_mid_point(line_edges, y):
    white_pixels = np.nonzero(line_edges[y, :])[0]
    white_pixels = merge_close_edges(white_pixels)
    if len(white_pixels) == 0:
        return None
    elif len(white_pixels) == 1:  
        if LANE_PIXELS is not None:                         
            white_pixels = np.nonzero(line_edges[y, :])[0]
            if white_pixels[0] > line_edges.shape[1]//2:
            #     x_coords_points = 0, white_pixels[0]
            # else:
            #     x_coords_points = white_pixels[0], line_edges.shape[1]
            # if white_pixels[0] > line_edges.shape[1]//2:
                x_coords_points = white_pixels[0]-LANE_PIXELS, white_pixels[0]
            else:
                x_coords_points = white_pixels[0], white_pixels[0]+LANE_PIXELS
        else:
            return None
    elif len(white_pixels) == 2 and (0 in white_pixels or line_edges.shape[1]-1 in white_pixels):    
        if 0 in white_pixels and white_pixels[-1] >= line_edges.shape[1]//2:
            x_coords_points = white_pixels[-1]-LANE_PIXELS, white_pixels[-1]
        if 0 in white_pixels and white_pixels[-1] < line_edges.shape[1]//2:
            return -np.inf
        if line_edges.shape[1]-1 in white_pixels and white_pixels[0] <= line_edges.shape[1]//2:
            x_coords_points = white_pixels[0], white_pixels[0]+LANE_PIXELS
        if line_edges.shape[1]-1 in white_pixels and white_pixels[0] > line_edges.shape[1]//2:
            return +np.inf
   
    elif len(white_pixels) >= 2:
        max_diff = float('-inf')
        max_diff_indices = None
        
        for i in range(len(white_pixels) - 1):
            diff = abs(white_pixels[i] - white_pixels[i+1])
            if diff > max_diff:
                max_diff = diff
                max_diff_indices = (i, i+1)
        x_coords_points = white_pixels[max_diff_indices[0]], white_pixels[max_diff_indices[1]]
        # x_coords_points = white_pixels[0], white_pixels[-1]
    else:
        x_coords_points = white_pixels[0], white_pixels[1]
    return x_coords_points

def computing_mid_pointS(line_edges, y, th_y=300, n_point=5):
    y_values = [int(x) for x in np.linspace(th_y, y, n_point)[:-1]]
    midpoints = []
    for y_act in y_values:
        x_coords_points = computing_mid_point(line_edges, y_act)
        if x_coords_points is not None and x_coords_points != +np.inf and x_coords_points != -np.inf:
            posm = y_act, (x_coords_points[1] + x_coords_points[0])//2
            midpoints.append(posm)
    return midpoints

def computing_delta(midpoints, th_straight=20):
    global prev_curvature
    next_point = midpoints[-1]
    midpoints = np.array(midpoints)

    x = midpoints[:, 1] 
    delta_x = next_point[1] - x

    mean_delta_x = np.mean(delta_x)
    print("\t ---------- \t")
    print('Sum of delta_x =', -mean_delta_x)
    
    if prev_curvature is not None:
        if (prev_curvature == 'left' and mean_delta_x < 0) or (prev_curvature == 'right' and mean_delta_x > 0):
            return prev_curvature

    if abs(mean_delta_x) < th_straight:
        curvature = 'straight'
    elif mean_delta_x > 0:
        curvature = 'left'
    elif mean_delta_x < 0:
        curvature = 'right'

    prev_curvature = curvature

    return curvature

def computing_lateral_distance(line_edges, show=False):
    global LANE_PIXELS
    global LATERAL_DISTANCE
    global scale_factor
    if prev_curvature is not None:
        if prev_curvature != "straight":
            y = Y_METERS[7.5]
            long_dist = 7.5
        else: 
            y = Y_METERS[7.5]
            long_dist = 7.5
    else:
        y = Y_METERS[7.5]
        long_dist = 7.5   
    x_coords_points = computing_mid_point(line_edges, y)

    if x_coords_points is None:
        return LATERAL_DISTANCE, long_dist, None
    if x_coords_points == -np.inf:
        return -np.inf, long_dist, None
    elif x_coords_points == np.inf:
        return np.inf, long_dist, None
    
    posm = y, (x_coords_points[1] + x_coords_points[0])//2

    middle_image = line_edges.shape[1]//2
    lateral_distance = posm[1] - middle_image
    if not LANE_PIXELS:
        LANE_PIXELS = x_coords_points[1] - x_coords_points[0]   
        scale_factor = LANE_METERS / LANE_PIXELS          
    
    later_distance_meters = lateral_distance * scale_factor
    LATERAL_DISTANCE = later_distance_meters

    
    midpoints = computing_mid_pointS(line_edges, y)
    midpoints.append(posm)
    if len(midpoints) > 1:
        curvature = computing_delta(midpoints)
    else:
        curvature = None

    if midpoints is not None:
        posm = midpoints[-1]
        midpoints = midpoints[:-1]
        cv2.circle(line_edges, tuple(posm[::-1]), 2, (255, 255, 255), 5)
        # for p in midpoints:
        #     cv2.circle(line_edges, tuple(p[::-1]), 2, (200, 200, 200), 3)

    return later_distance_meters, long_dist, curvature

# utils_model_
def preprocessing_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_shape = 640

    normalize = transforms.Normalize(
        mean=params.mean, std=params.std
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    h0, w0 = image.shape[:2]  
    r = resized_shape / max(h0, w0) 
    input_img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)


    (input_img, _), _, _ = letterbox((input_img, None), resized_shape, auto=True,
                                            scaleup=False)

    if use_cuda:
        input_tensor = transform(input_img).unsqueeze(0).cuda()
    else:
        input_tensor = transform(input_img).unsqueeze(0).cpu()   
    return input_tensor

def initialize_model():
    MULTICLASS_MODE: str = "multiclass"

    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales
    obj_list = params.obj_list
    seg_list = params.seg_list

    use_cuda=torch.cuda.is_available()

    #-----------------change on each computer-------------------------------------------------------------------------------------
    weights_path = find_weights('hybridnets.pth')
    #-----------------------------------------------------------------------------------------------------------------------------
    state_dict = torch.load(weights_path, map_location='cuda' if use_cuda else 'cpu')
    print(f"{use_cuda=}")

    seg_mode = MULTICLASS_MODE

    model = HybridNetsBackbone(compound_coef=3, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                            scales=eval(anchors_scales), seg_classes=len(seg_list), seg_mode=seg_mode)           # lasciare None sulla backbone è ok

    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    return model

def preprocessing_mask(seg, show=False,open=True, close=True ):
    _, seg_mask = torch.max(seg, 1)
    seg_mask_ = seg_mask[0].squeeze().cpu().numpy()
    pad_h = int(shapes[0][1][1][1])
    pad_w = int(shapes[0][1][1][0])
    seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
    seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[0][0][::-1], interpolation=cv2.INTER_NEAREST)
    color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
    for index, seg_class in enumerate(params.seg_list):
        if seg_class == 'road': # 'road', 'lane', or remove this line for both 'road' and 'lane'
            color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)

    _, end_mask = cv2.threshold(color_mask,0,255, cv2.THRESH_BINARY)
    _,labeled_image, stats, _ = cv2.connectedComponentsWithStats(image=np.uint8(end_mask))
    wanted_label=np.argmax(stats[1::,4])+1
    end_mask=np.array(np.where(labeled_image==wanted_label,255,0),dtype=np.uint8)
    if show:
        plt.imshow(end_mask)
        plt.show()
    return end_mask.astype('uint8')

# DEBUG
counter = 0

def set_debug_folders():
    debug_folder = os.path.join(os.getcwd(), "DEBUG")
    try:
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
    except OSError as e:
        print(f"Error creating debug folder: {e}")
    
    now = datetime.now ()
    folder = os.path.join(debug_folder, now.strftime (f"%Y_%m_%d_%H_%M_%S"))
    os.makedirs(folder)
    # try:
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    # except OSError as e:
    #     print(f"Error creating date folder: {e}")

    # Create subfolders for logs, output, and frames
    logs_folder = os.path.join(folder, "logs")
    output_folder = os.path.join(folder, "output")
    frames_folder = os.path.join(folder, "frames")
    os.makedirs(logs_folder)
    os.makedirs(output_folder)
    os.makedirs(frames_folder)
    # try:
    #     # Try to create the subfolders if they do not exist
    #     if not os.path.exists(logs_folder):
    #         os.makedirs(logs_folder)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     if not os.path.exists(frames_folder):
    #         os.makedirs(frames_folder)
    # except OSError as e:
    #     # Handle the error if the folder creation fails
    #     print(f"Error creating subfolders: {e}")
    return logs_folder, output_folder,frames_folder

if DEBUG:
    logs_folder, output_folder,frames_folder = set_debug_folders()

def image_callback(data):
    global steer_pub
    global counter
    global goal_published
    # convert ROS Image message to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    input_tensor = preprocessing_image(cv_image)
    with torch.no_grad():
        _, _, _, _, seg = model(input_tensor)
    mask = preprocessing_mask(seg, show=False,open=False, close=False)
    line_edges = processing_mask(mask, cv_image, show=False)

    '''file_path = os.path.expanduser("~/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/line_edges.txt")
    with open(file_path, "w") as file:
        for row in line_edges:
            file.write(' '.join([str(elem) for elem in row]) + '\n')'''

    lateral_distance, longitudinal_distance, curvature = computing_lateral_distance(line_edges, show=False)

        # Make sure line_edges is 3-channel like i4
    """if len(line_edges.shape) == 2:
        line_edges = cv2.cvtColor(line_edges, cv2.COLOR_GRAY2BGR)

        # Make sure mask is 3-channel like i4
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)"""

    resized_image = cv2.resize(cv_image, (200, 200))
    resized_mask = cv2.resize(mask, (200, 200))
    resized_line_edges = cv2.resize(line_edges, (200, 200))
    concatenated_image = np.hstack((resized_image, cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB), cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)))

    if lateral_distance == -np.inf:
        degree_steering_angle = -40.0
    elif lateral_distance == np.inf:
        degree_steering_angle = 40.0
    else:
        distance_to_waypoint = longitudinal_distance**2 + lateral_distance**2
        degree_steering_angle = math.degrees(math.atan2(2 * wheelbase * lateral_distance, distance_to_waypoint))  

    #cv2.imshow("RGB", resized_image)
    cv2.imshow("RGB, Mask and Line Edges", concatenated_image)
    cv2.waitKey(1)

    if DEBUG:
        resized_image = cv2.resize(cv_image, (540, 360))
        resized_mask = cv2.resize(mask, (540, 360))
        resized_line_edges = cv2.resize(line_edges, (540, 360))
        concatenated_image = np.hstack((cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB), cv2.cvtColor(resized_line_edges, cv2.COLOR_GRAY2BGR)))
        
        frame_name = f"frame_{counter}.png"
        frame_path = os.path.join(frames_folder, frame_name)
        cv2.imwrite(frame_path, cv_image)

        output_name = f"output_{counter}.png"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, concatenated_image)

        log_file = os.path.join(logs_folder, f"log_{counter}.txt")
        log = open(log_file,"w")

        #log.write(f"{counter}: Curvature: {curvature} - longitudinal_distance: {longitudinal_distance} - degree_steering_angle: {degree_steering_angle}\n")
        log.close()
        counter += 1
    
    print(f"{curvature = }")
    print(f"{longitudinal_distance = }")
    print(f"{lateral_distance = }")
    print(f"{degree_steering_angle = }")
    print("\t ---------- \t")
    
    goal_msg = PoseStamped()
    goal_msg.header.stamp = rospy.Time.now()
    goal_msg.header.frame_id = "ego_vehicle"
    goal_msg.pose.position.x = longitudinal_distance
    goal_msg.pose.position.y = -lateral_distance
    goal_msg.pose.position.z = 0.0
    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.0
    goal_msg.pose.orientation.w = 1

    #print(f"{goal_msg.pose.position.x = }")
    #print(f"{goal_msg.pose.position.y = }")

    goal_pub.publish(goal_msg)

if __name__ == '__main__':
    rospy.init_node('carla_path_planning_plus2', anonymous=True)
    bridge = CvBridge()
    model = initialize_model()
    
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    tf_listener = tf.TransformListener()
    rospy.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, image_callback)

    print("------------------Hello carla path planning------------------------------")

    rospy.spin()  # Keep the node running and listening for callbacks