#!/usr/bin/env python3

import numpy as np
import cv2
from progress.bar import FillingSquaresBar
import argparse
import matplotlib.pyplot as plt


oflow_params = dict(
                pyr_scale=0.5,
                levels=5, 
                winsize=10, 
                iterations=1, 
                poly_n=5, 
                poly_sigma=1.1, 
                flags=0
            )

cv2.namedWindow('Original')
cv2.namedWindow('Optical flow')

def get_parser():
    parser = argparse.ArgumentParser(description="Get optical flow for videos")
    parser.add_argument("-v","--video", help="Path to video file.")
    parser.add_argument("-s","--scale", help="Scale the video.", default=0.5, type=float)
    parser.add_argument("-b","--blur", help="Blur the video.", type=int, default=None)

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    

    if args.video:

        video = cv2.VideoCapture(args.video)

        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        DIM = int(args.scale*height)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # ret, frame_bgr = video.read()
        # frame = cv2.resize(frame_bgr, (DIM,DIM), interpolation = cv2.INTER_AREA)
        # prev_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # print(np.dtype(prev_frame[0,0]))
        hsv = np.zeros((DIM,DIM,3), dtype=np.uint8)
        hsv[:,:,1] = 255

        magnitude = []
        count_frames = 0
        aux = 0

        while(video.isOpened()):
            #print(count_frames)

            ret, frame_bgr = video.read()

            if ret:

                frame = cv2.resize(frame_bgr, (DIM,DIM), interpolation = cv2.INTER_AREA)
                gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                if args.blur is not None:
                    ksize = (int(args.blur), int(args.blur))
                    gray_frame = cv2.blur(gray_frame, ksize) 

                if count_frames > 0:
                    
                    flow = cv2.calcOpticalFlowFarneback(prev_frame,gray_frame, None, **oflow_params)
                    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
                    #print(np.linalg.norm(mag, ord='fro'))
                    norm = np.linalg.norm(mag, ord='fro')


                    if np.isinf(norm):
                        #print(norm)
                        magnitude.append(aux1)
                    else:
                        magnitude.append(norm)
                        aux1 = norm
                    # print(mag.shape)
                    # print(np.concatenate(mag).shape)

                    hsv[:,:,0] = ang*180/np.pi/2
                    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                    cv2.imshow('Original',gray_frame)
                    cv2.imshow('Optical flow',bgr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord('s'):
                        cv2.imwrite('opticalfb.png',frame)
                        cv2.imwrite('opticalhsv.png',flow[:,:,1])

                prev_frame = gray_frame
                count_frames += 1

            if count_frames >= num_frames:
                break

        cv2.destroyAllWindows()
        video.release()

        if len(magnitude) == num_frames-1:
            plt.plot(np.arange(0,num_frames-1)/30,magnitude)
            plt.show()