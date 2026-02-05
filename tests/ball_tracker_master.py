#!/usr/bin/env python3
"""
Master ball tracking test: works with a live camera or a prerecorded video.

Examples:
  python3 tests/ball_tracker_master.py --camera 0
  python3 tests/ball_tracker_master.py --video path/to/video.mp4
"""

import argparse
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master ball tracking test")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="Camera index, e.g., 0")
    src.add_argument("--video", type=str, help="Path to prerecorded video")

    parser.add_argument("--width", type=int, default=1280, help="Capture width (camera only)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (camera only)")
    parser.add_argument("--min-area", type=int, default=800, help="Minimum contour area for detection")
    parser.add_argument("--circularity", type=float, default=0.83, help="Min circularity (0-1)")
    parser.add_argument("--show-mask", action="store_true", help="Show threshold mask windows")
    parser.add_argument("--text", type=bool, default=False, help="Show the text of circularity and area")
    return parser.parse_args()


def build_masks(hsv: np.ndarray):
    # Red wraps around HSV, so we use two ranges.
    lower_red1 = np.array([170, 180, 70])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([170, 180, 70])
    upper_red2 = np.array([180, 255, 255])

    # White: low saturation, high value.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 120, 255])

    # Black: low value.
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    # Yellow (黄色): Hue 约为 25-35
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Green (绿色): Hue 约为 50-70
    lower_green = np.array([65, 220, 130])
    upper_green = np.array([80, 255, 255])

    # Blue (蓝色): Hue 约为 100-130
    lower_blue = np.array([80, 130, 180]) 
    upper_blue = np.array([120, 255, 255])

    # Pink (粉色): Hue 约为 145-165 (接近红色但偏紫)
    lower_pink = np.array([165, 100, 150]) # 饱和度可以稍低一点
    upper_pink = np.array([180, 200, 255])

    # Brown (棕色): 本质是暗橙色/暗红色. Hue 10-20, 但亮度(V)要低
    # 棕色是最难调的，因为它很容易和红色或阴影混淆
    lower_brown = np.array([10, 150, 120])  
    upper_brown = np.array([20, 255, 255]) # V 上限限制在 150，太亮就成橙色了

    masks = {}

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    masks['red'] = cv2.bitwise_or(mask1, mask2)

    masks['white'] = cv2.inRange(hsv, lower_white, upper_white)
    masks['black'] = cv2.inRange(hsv, lower_black, upper_black)
    masks['yellow'] = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masks['green']  = cv2.inRange(hsv, lower_green, upper_green)
    masks['blue']   = cv2.inRange(hsv, lower_blue, upper_blue)
    masks['pink']   = cv2.inRange(hsv, lower_pink, upper_pink)
    masks['brown']  = cv2.inRange(hsv, lower_brown, upper_brown)

    # 形态学操作 (去噪)
    kernel = np.ones((5, 5), np.uint8)
    for color in masks:
        masks[color] = cv2.morphologyEx(masks[color], cv2.MORPH_OPEN, kernel)
        masks[color] = cv2.morphologyEx(masks[color], cv2.MORPH_CLOSE, kernel)

    return masks


def find_ball_contours(mask: np.ndarray, min_area: int, min_circularity: float, check_isolation: bool = False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    # pre-create a black canvas, avoid creating it repeatedly in the loop, improve performance
    h, w = mask.shape
    blank_canvas = np.zeros((h, w), np.uint8)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
            
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
            
        # isolation check
        if check_isolation:
            # 1. draw the current outline (solid) on a blank canvas.
            # optimize speed by using local ROI (Region of Interest) to avoid full-map operations (optional, but use the full map here for simplicity).
            mask_current = np.zeros_like(mask)
            cv2.drawContours(mask_current, [cnt], -1, 255, -1)
            
            # 2. expand the outline outwards to form a larger area.
            # the kernel size determines how many pixels of surrounding data are detected.
            kernel = np.ones((200, 200), np.uint8) 
            mask_dilated = cv2.dilate(mask_current, kernel, iterations=1)
            
            # 3. subtracting the two yields the "ring-shaped region".
            mask_ring = cv2.bitwise_xor(mask_dilated, mask_current)
            
            # 4. calculate the total number of pixels in the annular region
            ring_total_pixels = cv2.countNonZero(mask_ring)
            
            if ring_total_pixels > 0:
                # 5. check how much of this ring-shaped area in the original mask is the target color?
                ring_color_pixels = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=mask_ring))
                
                # claculate the proportion of target color
                ratio = ring_color_pixels / ring_total_pixels
                
                # if ratio larger than 80%, ignore it.
                if ratio > 0.8: 
                    continue
        balls.append((cnt, circularity, area))
    return balls


def draw_ball(frame, cnt, circularity, area, color, center_color=None, text=False):
    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    cv2.circle(frame, center, int(radius), color, 2)
    cv2.circle(frame, center, 3, center_color or color, -1)


    if text:
        text_circ = f"Circ: {circularity:.2f}"
        text_area = f"Area: {int(area)}"

        # set font parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6       
        thickness_inner = 2    
        thickness_outer = 4    
        text_color = (0, 255, 0) # green
        
        # draw circularity
        cv2.putText(frame, text_circ, (int(x) + 10, int(y) - 10), 
                    font, font_scale, (0, 0, 0), thickness_outer)
        cv2.putText(frame, text_circ, (int(x) + 10, int(y) - 10), 
                    font, font_scale, text_color, thickness_inner) 

        # draw area
        cv2.putText(frame, text_area, (int(x) + 10, int(y) + 15), 
                    font, font_scale, (0, 0, 0), thickness_outer)
        cv2.putText(frame, text_area, (int(x) + 10, int(y) + 15), 
                    font, font_scale, (0, 255, 255), thickness_inner) # yellow


def main() -> None:
    args = parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video {args.video}")

    cv2.namedWindow("Ball Tracker", cv2.WINDOW_NORMAL)

    if args.show_mask:
        # 定义所有需要显示的颜色键值 (与 build_masks 里的 key 对应)
        all_colors = ['red', 'white', 'black', 'yellow', 'green', 'blue', 'pink', 'brown']
        
        for color in all_colors:
            window_name = f"{color.capitalize()} Mask" # 首字母大写，例如 "Red Mask"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # 强烈建议：设置一个小一点的默认尺寸，否则8个窗口弹出来会铺满屏幕
            cv2.resizeWindow(window_name, 320, 240)

    last_time = time.time()
    fps = 0.0

    colors_bgr = {
        'red':    (0, 0, 255),
        'white':  (255, 255, 255),
        'black':  (0, 0, 0),
        'yellow': (0, 255, 255),
        'green':  (0, 255, 0),
        'blue':   (255, 0, 0),
        'pink':   (180, 105, 255), # Hot Pink
        'brown':  (42, 42, 165)    # Brown
    }

    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # param 是我们在 setMouseCallback 中传入的 frame 变量的引用容器
            # 但由于 frame 在 while 循环中不断刷新，直接读取全局或外部变量比较方便
            # 这里为了简单，我们假设 'current_hsv' 是一个全局变量或者外部可访问的
            pixel = hsv[y, x]
            print(f"点击位置: ({x}, {y})")
            print(f"HSV值: H={pixel[0]}, S={pixel[1]}, V={pixel[2]}")
            print(f"建议范围: Lower=[0, 0, {max(0, pixel[2]-40)}], Upper=[180, {min(255, pixel[1]+30)}, 255]")
            print("-" * 30)

    cv2.setMouseCallback("Ball Tracker", pick_color)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or frame capture failed.")
            break

        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        global hsv 
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        masks = build_masks(hsv)

        # 2. 遍历字典，处理每一种颜色的球
        for color_name, mask in masks.items():
            
            # 针对白球开启特殊检查 (Isolation Check)，其他球通常不需要
            check_iso = (color_name == 'white')
            
            # 使用你之前修改好的 find_ball_contours (记得它现在返回三个值!)
            balls = find_ball_contours(mask, args.min_area, args.circularity, check_isolation=check_iso)
            
            # 绘制
            for cnt, circ, area in balls:
                # 如果是黑球，中心点画白色；其他球中心点画黑色或默认
                center_color = (255, 255, 255) if color_name == 'black' else None
                
                # 调用你优化过的 draw_ball
                draw_ball(frame, cnt, circ, area, colors_bgr[color_name], center_color)

        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Ball Tracker", frame)
        if args.show_mask:
            for color_name, mask_img in masks.items():
                window_name = f"{color_name.capitalize()} Mask"
                cv2.imshow(window_name, mask_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
