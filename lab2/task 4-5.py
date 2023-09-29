import cv2
import numpy as np


def cam_show():
    cap = cv2.VideoCapture(0)
    lower_red1 = np.array([0, 125, 85])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([167, 115, 75])
    upper_red2 = np.array([180, 255, 255])
    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        final_mask = cv2.addWeighted(mask1,0.5, mask2, 0.5, 0.0)
        red_filtered_frame = cv2.bitwise_and(frame, frame, mask=final_mask)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 300:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 0), -1)

        cv2.imshow('Red Filtered Image', red_filtered_frame)
        cv2.imshow('orig', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


cam_show()
