import cv2
import mediapipe as mp
import numpy as np

hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

with hands.Hands(min_detection_confidence=0.69, min_tracking_confidence=0.69) as hands_instance:
    while True:
        check, im = cam.read()
        if not check:
            print("Failed to grab frame")
            break

        im = cv2.flip(im, 1)
        rgb_frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        result = hands_instance.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = im.shape
                landmark_coords = []

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmark_coords.append((x, y))

                x_min = min([coord[0] for coord in landmark_coords])
                x_max = max([coord[0] for coord in landmark_coords])
                y_min = min([coord[1] for coord in landmark_coords])
                y_max = max([coord[1] for coord in landmark_coords])

                hand_roi = im[y_min:y_max, x_min:x_max]
                hsv_frame = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 48, 80], dtype="uint8")
                upper_skin = np.array([20, 255, 255], dtype="uint8")
                skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
                skin_mask = cv2.blur(skin_mask, (2, 2))
                contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=lambda x: cv2.contourArea(x))
                    cv2.drawContours(im[y_min:y_max, x_min:x_max], [largest_contour], -1, (0, 255, 0), 2)

                draw.draw_landmarks(
                    im,
                    hand_landmarks,
                    hands.HAND_CONNECTIONS
                )

        cv2.imshow('Hand Tracking with Hand-Only Skin Detection', im)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
