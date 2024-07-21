import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import speech_recognition as sr
import os
import signal
from screen_brightness_control import get_brightness, set_brightness

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return None

    try:
        text = recognizer.recognize_google(audio).lower()
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not hear your command.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Function to control audio based on hand gestures
def control_audio(mode):
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

            if mode == 'volume':
                length = hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [30, 350], [vol_min, vol_max])
                vol_bar = np.interp(length, [30, 350], [400, 150])

                print(vol, int(length))
                volume.SetMasterVolumeLevel(vol, None)

                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f"{int(np.interp(length, [30, 350], [0, 100]))}%",
                            (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
                
                thumb_status = detect_thumb_status(lm_list)
                if thumb_status == "thumb_up":
                    print("Unmuting Audio.")
                    volume.SetMute(False, None)
                    mute_status = False
                    # Implement bass increase logic
                elif thumb_status == "thumb_down":
                    print("Muting audio.")
                    volume.SetMute(True, None)
                    mute_status = True

            elif mode == 'brightness':
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

# Function to detect thumb status
def detect_thumb_status(hand_landmarks):
    whole_hand_present = all(isinstance(x, int) and isinstance(y, int) for _, x, y in hand_landmarks)

    if whole_hand_present:
        thumb_tip_y = hand_landmarks[4][2]
        return "thumb_up" if thumb_tip_y < hand_landmarks[2][2] else "thumb_down"
    else:
        return "other"

print("Listening for voice command...")
while True:
    command = recognize_speech()

    if command:
        if "volume" in command:
            print("Controlling Volume...")
            control_audio('volume')
        elif "brightness" in command:
            print("Adjusting Brightness...")
            control_audio('brightness')
        elif "exit" in command:
            print("Exiting...")
            os.kill(os.getpid(), signal.SIGINT)
            break
        else:
            print("Command not recognized. Try saying 'volume', 'brightness', or 'exit.'")
