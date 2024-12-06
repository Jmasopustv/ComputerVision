import cv2
import numpy as np

def initialize_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

def initialize_kalman():
    kf = cv2.KalmanFilter(4, 2)  
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1
    kf.statePost = np.zeros((4, 1), np.float32)
    return kf

def track_pedestrian():
    hog = initialize_hog()
    kf = initialize_kalman()
    for i in range(1, 160):  
        frame = cv2.imread(f'walking/{i:03d}.jpg')
        if frame is None:
            continue

        rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        
        if len(rects) > 0:
            rects = sorted(zip(rects, weights), key=lambda x: x[1], reverse=True)
            x, y, w, h = rects[0][0]
            cx, cy = x + w // 2, y + h // 2  

            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        
        prediction = kf.predict()
        px, py = int(prediction[0][0]), int(prediction[1][0]) 

        cv2.rectangle(frame, (px - 15, py - 15), (px + 15, py + 15), (0, 255, 0), 2)
        cv2.imshow('Pedestrian Tracking', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

track_pedestrian()

