import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Function to control audio based on hand gestures
def control_audio():
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol_bar = 400
    vol_min, vol_max = volume.GetVolumeRange()[:2]

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
                lm_list.extend([(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0]))
                                for id, lm in enumerate(hand_landmark.landmark)])

                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

        if lm_list:
            x1, y1 = lm_list[4][1:3]
            x2, y2 = lm_list[8][1:3]

            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Control volume based on hand gestures
            length = hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [30, 350], [vol_min, vol_max])
            vol_bar = np.interp(length, [30, 350], [400, 150])

            print(vol, int(length))
            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(np.interp(length, [30, 350], [0, 100]))}%",
                        (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)

        cv2.imshow('Hand Gestures', img)

        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Controlling Audio based on Hand Gestures...")
control_audio()
