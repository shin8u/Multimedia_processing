import cv2
import numpy as np

i = 0

def main(kernel_size, standard_deviation, delta_tresh, min_area):
    global i
    i += 1

    video = cv2.VideoCapture(r'.\LR5.mp4', cv2.CAP_ANY)

    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(r'.\output ' + str(i) + '.mp4', fourcc, 144, (w, h))
    j = 0
    while True:
        old_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)


        diff = cv2.absdiff(img, old_img)
        thresh = cv2.threshold(diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        (contors, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('wind1', frame)
        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video_writer.release()




kernel_size = 11
standard_deviation = 60
delta_tresh = 100
min_area = 10
main(kernel_size, standard_deviation, delta_tresh, min_area)



