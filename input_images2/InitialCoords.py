import cv2
import numpy as np

frame = cv2.imread('pres_debate_noisy/000.jpg') 

roi = cv2.selectROI("Select Hand", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

initial_pos = (int(roi[0] + roi[2] / 2), int(roi[1] + roi[3] / 2))

print("Initial Position:", initial_pos)

