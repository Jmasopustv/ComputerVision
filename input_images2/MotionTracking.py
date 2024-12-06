import cv2
import numpy as np

def initialize_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    
    kf.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)  # Q matrix
    kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)  # R matrix
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[0], [0], [0], [0]], np.float32)

    return kf

def find_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=20, minRadius=10, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0] 
    return None

def process_frame(kf, frame, measurement):
    if measurement is not None:
        kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))
    prediction = kf.predict()
    return prediction

def track_circle():
    kf = initialize_kalman()
    last_measurement = None

    for i in range(100):
        frame = cv2.imread(f'circle/{i:04d}.jpg')
        if frame is None:
            continue

        current_measurement = find_circle(frame)

        if current_measurement is not None:
            kf.correct(np.array([[np.float32(current_measurement[0])], [np.float32(current_measurement[1])]]))
            last_measurement = current_measurement
        else:
            if last_measurement is not None:
                kf.correct(np.array([[np.float32(last_measurement[0])], [np.float32(last_measurement[1])]]))

        prediction = kf.predict() 
        x, y = int(prediction[0][0]), int(prediction[1][0])

        cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)  
        cv2.imshow('Circle Tracking', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

track_circle()


