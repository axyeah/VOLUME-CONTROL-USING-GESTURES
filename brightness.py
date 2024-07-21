import cv2
import mediapipe as mp
from math import hypot
from screen_brightness_control import get_brightness, set_brightness

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Function to recognize hand gestures and adjust brightness
def adjust_brightness():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Error: Empty frame.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        lm_list = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                lm_list.extend([(lm.x * img.shape[1], lm.y * img.shape[0])
                                for lm in hand_landmark.landmark])

                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

                # Get the positions of the pinky and thumb
                pinky_tip = (int(lm_list[20][0]), int(lm_list[20][1]))
                thumb_tip = (int(lm_list[4][0]), int(lm_list[4][1]))

                # Calculate distance between pinky and thumb
                distance = calculate_distance(pinky_tip, thumb_tip)

                # Adjust brightness based on the distance
                brightness_change = int((distance - 50) / 10)
                current_brightness = get_brightness(display=0)[0]  # Extract the brightness value from the list
                new_brightness = max(0, min(100, current_brightness + brightness_change))

                # Set the new brightness
                set_brightness(new_brightness, display=0)

        cv2.imshow('Hand Gestures', img)

        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the brightness adjustment function
adjust_brightness()
