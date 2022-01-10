import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dev", default=0, help="ID of usb camera device")
args = parser.parse_args()

cap = cv2.VideoCapture(int(args.dev))

# 設定影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    
    t_start=time.time()

    ret, frame = cap.read()

    cv2.putText(frame, f'FPS: {int(1/(time.time()-t_start))}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Testing Web Cam', frame)
    
    key = cv2.waitKey(1)
    if key==ord('q') or key==27: break

cap.release()
cv2.destroyAllWindows()
print('1')