import cv2
import numpy as np;
import dlib
from math import hypot
import pyautogui
def get_horizontal_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray_scale_frame, gray_scale_frame, mask=mask)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

def get_vertical_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray_scale_frame, gray_scale_frame, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 100, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height//2, 0: width]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[height//2: height,0: width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
        left_side_threshold = cv2.resize(left_side_threshold,None,fx=8,fy=8)
        right_side_threshold = cv2.resize(right_side_threshold, None, fx=8, fy=8)
        cv2.imshow("top",left_side_threshold)
        cv2.imshow("bottom",right_side_threshold)
    return gaze_ratio
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def midpoint(p1,p2):
    return (int((p1.x+p2.x)/2), int((p1.y+p2.y)/2))

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
x,y=0,0
pyautogui.FAILSAFE=False
speed=19
while True:
    ret,frame = cap.read()
    gray_scale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_scale_frame)
    for face in faces:
        #draw rect
        x1,y1=face.left(),face.top()
        x2,y2=face.right(),face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),[0,255,0],2)
        landmarks = predictor(gray_scale_frame,face)
        #cv2.circle(frame,(landmarks.part(36).x,landmarks.part(36).y),1,[0,0,255],1)
        #cv2.circle(frame, (landmarks.part(39).x, landmarks.part(39).y), 1, [0, 0, 255], 1)
        #her_line = cv2.line(frame,(landmarks.part(36).x,landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y),[0,255,0],2)
        #cv2.circle(frame, midpoint(landmarks.part(37),landmarks.part(38)), 1, [0, 0, 255], 1)
        #cv2.circle(frame, midpoint(landmarks.part(40),landmarks.part(41)), 1, [0, 0, 255], 1)
        #ver_line = cv2.line(frame, midpoint(landmarks.part(37),landmarks.part(38)), midpoint(landmarks.part(40),landmarks.part(41)),[0, 255, 0], 2)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        if(left_eye_ratio>5.6):
            cv2.putText(frame, "L - BLINKING", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0))
            pyautogui.click()
        #if blinking_ratio > 5.7:
         #   cv2.putText(frame, "BLINKING", (50, 150),cv2.FONT_HERSHEY_SIMPLEX, 7, (255, 0, 0))
        # left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
        #                             (landmarks.part(37).x,landmarks.part(37).y),
        #                             (landmarks.part(38).x,landmarks.part(38).y),
        #                             (landmarks.part(39).x,landmarks.part(39).y),
        #                             (landmarks.part(40).x,landmarks.part(40).y),
        #                             (landmarks.part(41).x,landmarks.part(41).y)])
        # right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
        #                              (landmarks.part(43).x, landmarks.part(43).y),
        #                              (landmarks.part(44).x, landmarks.part(44).y),
        #                              (landmarks.part(45).x, landmarks.part(45).y),
        #                              (landmarks.part(46).x, landmarks.part(46).y),
        #                              (landmarks.part(47).x, landmarks.part(47).y)])
        # cv2.polylines(frame,[left_eye_region],True,[0,255,0],1)
        # cv2.polylines(frame, [right_eye_region], True, [0, 255, 0], 1)

        horizontal_gaze_ratio_left_eye = get_horizontal_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        horizontal_gaze_ratio_right_eye = get_horizontal_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        vertical_gaze_ratio_left_eye = get_vertical_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        vertical_gaze_ratio_right_eye = get_vertical_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        vertical_gaze_ratio = (vertical_gaze_ratio_left_eye + vertical_gaze_ratio_right_eye) / 2
        horizontal_gaze_ratio = (horizontal_gaze_ratio_right_eye + horizontal_gaze_ratio_left_eye) / 2
        print(vertical_gaze_ratio_left_eye)
        if horizontal_gaze_ratio <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            x+=speed

        elif 1 < horizontal_gaze_ratio < 1.7:
            cv2.putText(frame, "CENTER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            x=x

        else:
            cv2.putText(frame, "LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            x-=speed
        cv2.putText(frame, str(vertical_gaze_ratio), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        if vertical_gaze_ratio <0.4:
            cv2.putText(frame, "Top", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            y-=speed
        elif vertical_gaze_ratio>=1:
            cv2.putText(frame, "Bottom", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            y+=speed
        else:
            y=y


        pyautogui.moveTo(x,y)
    cv2.imshow("Frame",frame)
    cv2.imshow("GrayFrame",gray_scale_frame)
    key = cv2.waitKey(27)
    if(key==27):
        break
cap.release()
cv2.destroyAllWindows()
