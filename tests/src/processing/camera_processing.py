import cv2
from cv2 import aruco
import os
import glob
import json
import logging
import numpy as np

# Import the config instance directly
from src.core import config

logger=  logging.getLogger(__name__)

def get_top_down_view(frame, homography_matrix):
    top_down_view = cv2.warpPerspective(
        frame, 
        homography_matrix, 
        config.output_dimensions, 
        borderMode=cv2.BORDER_REPLICATE)
    
    return cv2.flip(top_down_view, 0)


def use_calibration(frame):
    if os.path.exists(config.calibration_params_path):
        logger.info(
            "Camera calibration parameters already exist, skipping calibration.")
        
        with open(config.calibration_params_path, 'r') as f:
            calibration_params = json.load(f)

        mtx = np.array(calibration_params['mtx'])
        dist = np.array(calibration_params['dist'])
        h, w = frame.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        return mtx, dist, newcameramtx, roi
    
    logger.info("Calibration data not found, calibrating camera.")

    # Define ChArUco board
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((6,8), 34, 27, aruco_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Get images
    images = glob.glob(config.calibration_images_path + "*.jpg")
    image = cv2.imread(images[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_size = image.shape

    all_charuco_ids, all_charuco_corners = get_charuco_corners_and_ids(images, detector, board)

    _, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, 
        all_charuco_ids, 
        board, 
        img_size, 
        None, None)

    data = {
        'mtx': mtx.tolist(),
        'dist': dist.tolist()
    }

    with open(config.calibration_params_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info("Camera calibration parameters saved.")
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return mtx, dist, newcameramtx, roi


def get_charuco_corners_and_ids(images, detector, board):
    all_charuco_ids = []
    all_charuco_corners = []

    for image_file in images:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if len(marker_ids) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, 
                marker_ids, 
                gray, 
                board)
            
            if ret > 0:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
    
    return all_charuco_ids, all_charuco_corners


def handle_calibration(frame):
    mtx, dist, newcameramtx, roi = None, None, None, None
    if not config.use_calibration:
        return mtx, dist, newcameramtx, roi
    
    if not os.path.exists(config.calibration_images_path):
        logger.error("Calibration folder does not exist.")
        return mtx, dist, newcameramtx, roi
    
    mtx, dist, newcameramtx, roi = use_calibration(frame)
    return mtx, dist, newcameramtx, roi


def undistort_frame(frame, mtx, dist, newcameramtx, roi):
    if not config.use_calibration:
        return frame
    
    if frame is None or mtx is None or dist is None:
        logging.warning("Frame, mtx, or dist is none. Cannot undistort.")
        return frame
    
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop the image to the ROI
    x, y, w, h = roi
    return undistorted_frame[y:y+h, x:x+w]


def select_points(event, x, y, _, param):
    table_pts = param

    if event == cv2.EVENT_LBUTTONDOWN and len(table_pts) < 4:
        table_pts.append((x, y))
        logger.info(f"Point selected: {x}, {y}")
        

def load_table_pts():
    if not os.path.exists(config.table_pts_path):
        logger.warning(
            f"{config.table_pts_path} does not exist. Please select table points.")
        return None
    
    with open(config.table_pts_path, 'r') as f:
        data = json.load(f)

    try:
        return np.array(data['table_pts'], dtype=np.float32)
    except KeyError:
        logger.error("Invalid table points format. Please select table points again.")
        return None


def save_table_pts(table_pts):
    data = {
        "table_pts": [list(pt) for pt in table_pts]
    }

    try:
        os.makedirs(os.path.dirname(config.table_pts_path), exist_ok=True)
        with open(config.table_pts_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Table points saved to {config.table_pts_path}.")
    except Exception as e:
        logger.error(f"Error saving table points: {e}")


def manage_point_selection(frame):
    sorted_pts = load_table_pts()
    if sorted_pts is None:
        table_pts = []

        cv2.namedWindow("Select Table Points")
        cv2.setMouseCallback("Select Table Points", select_points, table_pts)
        
        logger.info("Select 4 points.")

        while True:
            display_frame = frame.copy()
            for pt in table_pts:
                cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(
                    display_frame, 
                    str(pt), 
                    (pt[0] + 10, pt[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    config.font_scale, 
                    config.font_color, 
                    config.font_thickness)

            if len(table_pts) < 4:
                for i in range(1, len(table_pts)):
                    cv2.line(
                        display_frame, 
                        table_pts[i - 1], 
                        table_pts[i], 
                        (0, 0, 255), 
                        2)

            if len(table_pts) == 4:
                sorted_pts = sort_points(table_pts)
                cv2.line(display_frame, sorted_pts[0], 
                         sorted_pts[1], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[1], 
                         sorted_pts[3], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[3], 
                         sorted_pts[2], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[2], 
                         sorted_pts[0], (0, 0, 255), 2)

                cv2.putText(
                    display_frame, 
                    "Press Enter to confirm points", 
                    (frame.shape[1] // 2 - 150, frame.shape[0] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    config.font_scale, 
                    config.font_color, 
                    config.font_thickness)
            
            cv2.imshow("Select Table Points", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if (key == ord('\n') or key == ord('\r')) and len(table_pts) == 4:
                break
            if key == ord('\b') and len(table_pts) > 0:
                logger.info(f"Point {table_pts[-1]} removed.")
                table_pts.pop()
            if key == ord('q'):
                logger.info("Point selection cancelled.")
                cv2.destroyAllWindows()
                return None
            
        sorted_pts = sort_points(table_pts)
        save_table_pts(sorted_pts)
        cv2.destroyWindow("Select Table Points")

    return np.array(sorted_pts, dtype=np.float32)


def sort_points(table_pts):
    table_pts = sorted(table_pts, key=lambda x : x[1])
    top_pts = sorted(table_pts[:2], key=lambda x : x[0])
    bottom_pts = sorted(table_pts[2:], key=lambda x : x[0])
    return [top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]]



