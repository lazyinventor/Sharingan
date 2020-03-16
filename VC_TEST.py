import cv2
import numpy as np;
import dlib
from math import hypot
def midpoint(p1,p2):
    """Mid point for two points"""
    return (int((p1.x+p2.x)/2),int((p1.y+p2.y)/2))

def eye_trace(startpt,endpt,landmarks):
    """Trace the structure between the points"""
    for i in range(startpt,endpt):
        #cv2.circle(frame,(landmarks.part(i).x,landmarks.part(i).y),1,(0,0,255),2)
        pass

clearence=40
cap= cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    ret,frame=cap.read()
    frame = cv2.flip(frame,1)
    grayScale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(grayScale)
    for face in faces:
            landmarks=predictor(grayScale,face)
            #plotting  left 36-41
            eye_trace(36,42,landmarks)
            #cv2.line(frame,((landmarks.part(36).x,landmarks.part(36).y)),(landmarks.part(39).x,landmarks.part(39).y),[0,255,0],2)
            center_top_l= midpoint(landmarks.part(37),landmarks.part(38))
            center_bottom_l = midpoint(landmarks.part(40),landmarks.part(41))
            #cv2.line(frame,center_top_l,center_bottom_l,(0,255,0),2)
            
            hor_line_len_l =hypot((landmarks.part(36).x - landmarks.part(39).x),(landmarks.part(36).y-landmarks.part(39).y))
            ver_line_len_l = hypot(center_top_l[0]-center_bottom_l[0],center_top_l[1]-center_bottom_l[1])
            ratio_left = hor_line_len_l/ver_line_len_l
            
            #plotting  right 42-47
            eye_trace(42,48,landmarks)
            cv2.line(frame,((landmarks.part(42).x,landmarks.part(42).y)),(landmarks.part(45).x,landmarks.part(45).y),[0,255,0],2)
            center_topr= midpoint(landmarks.part(43),landmarks.part(44))
            center_bottomr =midpoint(landmarks.part(47),landmarks.part(46))
            cv2.line(frame,center_topr,center_bottomr,(0,255,0),2)
            hor_line_len_r =hypot((landmarks.part(42).x - landmarks.part(45).x),(landmarks.part(42).y-landmarks.part(45).y))
            ver_line_len_r = hypot(center_topr[0]-center_bottomr[0],center_topr[1]-center_bottomr[1])
            ratio_right = hor_line_len_r/ver_line_len_r

            if((ratio_left+ratio_right)/2>5.5):
                cv2.putText(frame,"Blinking",(50,150),cv2.FONT_HERSHEY_SIMPLEX,3,())

            left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
                                        (landmarks.part(37).x,landmarks.part(37).y),
                                        (landmarks.part(38).x,landmarks.part(38).y),
                                        (landmarks.part(39).x,landmarks.part(39).y),
                                        (landmarks.part(40).x,landmarks.part(40).y),
                                        (landmarks.part(41).x,landmarks.part(41).y)])

            right_eye_region = np.array([(landmarks.part(42).x,landmarks.part(42).y),
                                        (landmarks.part(43).x,landmarks.part(43).y),
                                        (landmarks.part(44).x,landmarks.part(44).y),
                                        (landmarks.part(45).x,landmarks.part(45).y),
                                        (landmarks.part(46).x,landmarks.part(46).y),
                                        (landmarks.part(47).x,landmarks.part(47).y)])

            #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
            #print(left_eye_region)
            min_x = np.min(left_eye_region[: ,0])
            max_x = np.max(left_eye_region[: ,0])
            min_y = np.min(left_eye_region[: ,1])
            max_y = np.max(left_eye_region[: ,1])
            eye=frame[min_y:max_y , min_x:max_x]
            
            min_xr = np.min(right_eye_region[: ,0])
            max_xr = np.max(right_eye_region[: ,0])
            min_yr = np.min(right_eye_region[: ,1])
            max_yr = np.max(right_eye_region[: ,1])
            reye=frame[min_yr:max_yr , min_xr:max_xr]

            rgray_eye = cv2.cvtColor(reye,cv2.COLOR_BGR2GRAY)
            lgray_eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
           # gray_eye = cv2.GaussianBlur(gray_eye,(1,1),0)

            _,thershold = cv2.threshold(lgray_eye,clearence,255,cv2.THRESH_BINARY)
            _=thersholdr = cv2.threshold(rgray_eye,clearence,255,cv2.THRESH_BINARY)

            eye=cv2.resize(eye,None,fx=10,fy=10)
            reye=cv2.resize(reye,None,fx=10,fy=10)

            thershold=cv2.resize(thershold,None,fx=10,fy=10)
            #thersholdr=cv2.resize(thersholdr,None,fx=10,fy=10)

            heiht,width,_=frame.shape
            mask = np.zeros((heiht,width),np.uint8)
           
            cv2.polylines(mask,[left_eye_region],True,255,2)
            cv2.fillPoly(mask,[left_eye_region],255)
            cv2.polylines(mask,[right_eye_region],True,255,2)
            cv2.fillPoly(mask,[right_eye_region],255)
            left_eye = cv2.bitwise_and(grayScale,grayScale,mask=mask)

            left_eye_frame = left_eye[min_y:max_y , min_x:max_x]
            right_eye_frame = left_eye[min_yr:max_yr , min_xr:max_xr]
            _,Lthershold = cv2.threshold(left_eye_frame,clearence,255,cv2.THRESH_BINARY)
            h,w = Lthershold.shape
            Lleft_side_ther = Lthershold[0:h,0:int(w/2)]
            Lright_side_ther = Lthershold[0:h,int(w/2):width]
            Lleft_side_ther=cv2.resize(Lleft_side_ther,None,fx=10,fy=10)
            Lright_side_ther=cv2.resize(Lright_side_ther,None,fx=10,fy=10)
            L_left_side_one=np.count_nonzero(Lleft_side_ther)
            L_right_side_one=np.count_nonzero(Lright_side_ther)

            cv2.putText(frame,"L:"+str(L_left_side_one),(50,170),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
            cv2.putText(frame,"R:"+str(L_right_side_one),(50,250),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)


            _,Rthershold = cv2.threshold(right_eye_frame,clearence,255,cv2.THRESH_BINARY)
            left_eye_frame=cv2.resize(left_eye_frame,None,fx=10,fy=10)
            Lthershold=cv2.resize(Lthershold,None,fx=10,fy=10)
            right_eye_frame=cv2.resize(right_eye_frame,None,fx=10,fy=10)
            Rthershold=cv2.resize(Rthershold,None,fx=10,fy=10)
            cv2.imshow("Lmask",left_eye)
            cv2.imshow("Frame",frame)
            cv2.imshow("left eye",left_eye_frame)
            cv2.imshow("L_left_side",Lleft_side_ther)
            cv2.imshow("R_left_side",Lright_side_ther)
            #cv2.imshow("left_eye_Thershold",Lthershold)
            #cv2.imshow("right eye",right_eye_frame)
            #cv2.imshow("right_eye_Thershold",Rthershold)
           # cv2.imshow("FrameGray",grayScale)



    key = cv2.waitKey(1)
    if(key==27):
        break
cap.release()
cv2.destroyAllWindows()
