import sys
import numpy as np
import cv2
import time
import pandas as pd

sys.path.insert(1, '../pyKinectAzure/')
# from kinectBodyTracker import kinectBodyTracker, _k4abt
from pyKinectAzure import pyKinectAzure, _k4a

# Path to the module
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll'
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'


# Transfer points from Azure Kinect space to OpenSim space
def transfer_points(x, y, z):
    joint_point = np.array([x, y, z])

    alpha = 90
    rotation_matrix_y = np.array([[np.cos(alpha * (np.pi / 180)), 0, np.sin(alpha * (np.pi / 180))],
                                  [0, 1, 0],
                                  [-np.sin(alpha * (np.pi / 180)), 0, np.cos(alpha * (np.pi / 180))]])
    rotated_point = np.dot(joint_point, rotation_matrix_y)

    alpha = 180
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(alpha * (np.pi / 180)), -np.sin(alpha * (np.pi / 180))],
                                  [0, np.sin(alpha * (np.pi / 180)), np.cos(alpha * (np.pi / 180))]])

    rotated_point = np.dot(rotated_point, rotation_matrix_x)
    rotated_point = np.add(rotated_point, 700)
    return rotated_point


# Waiting 3 seconds before start recording
def waiting_before_start():
    # Wait 3 seconds before starting the camera
    image3 = cv2.imread('C:/Users/Med/pyKinectAzure/321 images/3.png')
    image3 = cv2.resize(image3, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow("Segmented Depth Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Segmented Depth Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Segmented Depth Image', image3)
    cv2.waitKey(1)
    time.sleep(1)

    image2 = cv2.imread('C:/Users/Med/pyKinectAzure/321 images/2.png')
    cv2.imshow('Segmented Depth Image', image2)
    cv2.waitKey(1)
    time.sleep(1)

    image1 = cv2.imread('C:/Users/Med/pyKinectAzure/321 images/1.png')
    cv2.imshow('Segmented Depth Image', image1)
    cv2.waitKey(1)
    time.sleep(1)


if __name__ == "__main__":

    # Wait 3 seconds before starting the camera
    waiting_before_start()

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Initialize more things
    excel_file = 'C:/Users/Med/pyKinectAzure/examples/kinect-skeleton.xlsx'
    df = pd.read_excel(excel_file)
    curr_index = 5
    k = 0

    # Video Creation
    frames = []
    out = cv2.VideoWriter('Recorded.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 576))

    # Open device
    pyK4A.device_open()

    # Modify camera configuration
    device_config = pyK4A.config
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
    print(device_config)

    # Start cameras using modified configuration
    pyK4A.device_start_cameras(device_config)
    # Initialize the body tracker
    pyK4A.bodyTracker_start(bodyTrackingModulePath)

    # start real world counter
    start = time.time()

    while True:
        # Get capture
        pyK4A.device_get_capture()

        # Get the depth image from the capture
        depth_image_handle = pyK4A.capture_get_depth_image()

        if depth_image_handle:

            # Perform body detection
            pyK4A.bodyTracker_update()

            # Read and convert the image data to numpy array:
            depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
            depth_color_image = cv2.convertScaleAbs(depth_image,
                                                    alpha=0.05)  # alpha is fitted by visual comparison with Azure k4aviewer results
            depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB)

            # Get body segmentation image
            body_image_color = pyK4A.bodyTracker_get_body_segmentation()

            combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)
            K4ABT_JOINT_NAMES = ["pelvis", "spine - navel", "spine - chest", "neck", "left clavicle", "left shoulder",
                                 "left elbow",
                                 "left wrist", "left hand", " left hand-tip", "left thumb", "right clavicle",
                                 "right shoulder", "right elbow",
                                 "right wrist", "right hand", "right hand-tip", "right thumb", "left hip", "left knee",
                                 "left ankle", "right foot",
                                 "right hip", "right knee", "right ankle", "left foot", "head", "nose", "left eye",
                                 "left ear", "right eye", "right ear"]

            my_markers = ["neck", "right shoulder", "right shoulder", "right elbow", "right wrist",
                          "left shoulder", "left elbow", "left wrist", "pelvis", "right hip",
                          "left hip", "right knee", "right ankle", "left knee", "left ankle", "right foot", "left foot"]

            for body in pyK4A.body_tracker.bodiesNow:
                skeleton = body.skeleton
                print("-------------------", len(frames), "-------------------")
                for joint_idx, joint in enumerate(skeleton.joints):
                    curr_marker = K4ABT_JOINT_NAMES[joint_idx]
                    if curr_marker in my_markers and curr_index <= 500:
                        x_value = np.round(joint.position.v[0], 3)
                        y_value = np.round(joint.position.v[1], 3)
                        z_value = np.round(joint.position.v[2], 3)
                        OpenSim_Point = transfer_points(x_value, y_value, z_value)
                        if curr_marker == "neck":
                            stop = time.time()
                            df.iat[curr_index, 1] = np.round(stop - start, 3)
                            print("Time =", np.round(stop - start, 3))
                            df.iat[curr_index, 2] = OpenSim_Point[0]
                            df.iat[curr_index, 3] = OpenSim_Point[1]
                            df.iat[curr_index, 4] = OpenSim_Point[2]
                            df.iat[curr_index, 5] = OpenSim_Point[0]
                            df.iat[curr_index, 6] = OpenSim_Point[1]
                            df.iat[curr_index, 7] = OpenSim_Point[2]
                        elif curr_marker == "right shoulder":
                            df.iat[curr_index, 8] = OpenSim_Point[0]
                            df.iat[curr_index, 9] = OpenSim_Point[1]
                            df.iat[curr_index, 10] = OpenSim_Point[2]
                        elif curr_marker == "left shoulder":
                            df.iat[curr_index, 11] = OpenSim_Point[0]
                            df.iat[curr_index, 12] = OpenSim_Point[1]
                            df.iat[curr_index, 13] = OpenSim_Point[2]
                        elif curr_marker == "right elbow":
                            df.iat[curr_index, 14] = OpenSim_Point[0]
                            df.iat[curr_index, 15] = OpenSim_Point[1]
                            df.iat[curr_index, 16] = OpenSim_Point[2]
                        elif curr_marker == "left elbow":
                            df.iat[curr_index, 17] = OpenSim_Point[0]
                            df.iat[curr_index, 18] = OpenSim_Point[1]
                            df.iat[curr_index, 19] = OpenSim_Point[2]
                        elif curr_marker == "right wrist":
                            df.iat[curr_index, 20] = OpenSim_Point[0]
                            df.iat[curr_index, 21] = OpenSim_Point[1]
                            df.iat[curr_index, 22] = OpenSim_Point[2]
                        elif curr_marker == "left wrist":
                            df.iat[curr_index, 23] = OpenSim_Point[0]
                            df.iat[curr_index, 24] = OpenSim_Point[1]
                            df.iat[curr_index, 25] = OpenSim_Point[2]
                        elif curr_marker == "pelvis":
                            df.iat[curr_index, 26] = OpenSim_Point[0]
                            df.iat[curr_index, 27] = OpenSim_Point[1]
                            df.iat[curr_index, 28] = OpenSim_Point[2]
                        elif curr_marker == "right hip":
                            df.iat[curr_index, 29] = OpenSim_Point[0]
                            df.iat[curr_index, 30] = OpenSim_Point[1]
                            df.iat[curr_index, 31] = OpenSim_Point[2]
                        elif curr_marker == "left hip":
                            df.iat[curr_index, 32] = OpenSim_Point[0]
                            df.iat[curr_index, 33] = OpenSim_Point[1]
                            df.iat[curr_index, 34] = OpenSim_Point[2]
                        elif curr_marker == "right knee":
                            df.iat[curr_index, 35] = OpenSim_Point[0]
                            df.iat[curr_index, 36] = OpenSim_Point[1]
                            df.iat[curr_index, 37] = OpenSim_Point[2]
                        elif curr_marker == "left knee":
                            df.iat[curr_index, 38] = OpenSim_Point[0]
                            df.iat[curr_index, 39] = OpenSim_Point[1]
                            df.iat[curr_index, 40] = OpenSim_Point[2]
                        elif curr_marker == "right ankle":
                            df.iat[curr_index, 41] = OpenSim_Point[0]
                            df.iat[curr_index, 42] = OpenSim_Point[1]
                            df.iat[curr_index, 43] = OpenSim_Point[2]
                        elif curr_marker == "left ankle":
                            df.iat[curr_index, 44] = OpenSim_Point[0]
                            df.iat[curr_index, 45] = OpenSim_Point[1]
                            df.iat[curr_index, 46] = OpenSim_Point[2]
                        elif curr_marker == "right foot":
                            df.iat[curr_index, 50] = OpenSim_Point[0]
                            df.iat[curr_index, 51] = OpenSim_Point[1]
                            df.iat[curr_index, 52] = OpenSim_Point[2]
                        elif curr_marker == "left foot":
                            df.iat[curr_index, 47] = OpenSim_Point[0]
                            df.iat[curr_index, 48] = OpenSim_Point[1]
                            df.iat[curr_index, 49] = OpenSim_Point[2]

                curr_index += 1

            # Draw the skeleton
            for body in pyK4A.body_tracker.bodiesNow:
                skeleton2D = pyK4A.bodyTracker_project_skeleton(body.skeleton)
                combined_image = pyK4A.body_tracker.draw2DSkeleton(skeleton2D, body.id, combined_image)

            cv2.imshow('Segmented Depth Image', combined_image)
            frames.append(combined_image)

            k = cv2.waitKey(1)

            # Release the image
            pyK4A.image_release(depth_image_handle)
            pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)

        pyK4A.capture_release()
        pyK4A.body_tracker.release_frame()

        if k == 27 or curr_index >= 500:  # Esc key to stop or frames is full
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()
            break

    df.to_excel("output.xlsx", header=False, index=False)
    pyK4A.device_stop_cameras()
    pyK4A.device_close()
