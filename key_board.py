import cv2
import mediapipe as mp
from time import sleep
import numpy as np

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L","'"],
        ["Z","X","C","V","B","N","M",",",".","/"],
        ["Erase"]]

finalText = ""

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        # Draw a more creative button (e.g., rounded corners)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), -1)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                    4, (255, 255, 255), 4)
    return img

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        size = [150, 85] if key == "Erase" else [85, 85]
        buttonList.append(Button([j * 100 + 50, 100 * i + 50], key, size))

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lmList = [[lm.x * img.shape[1], lm.y * img.shape[0]] for lm in hand_landmarks.landmark]
            
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), -1)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)
                    
                    l = np.linalg.norm(np.array(lmList[8]) - np.array(lmList[12]))
                    
                    # When Clicked
                    if 45 < l < 50:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), -1)
                        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                                    4, (255, 255, 255), 4)
                        if button.text == "Erase":
                            finalText = ""
                        else:
                            finalText += button.text
                        sleep(0.25)
    
    img = drawAll(img, buttonList)

    cv2.rectangle(img, (50, 350), (1032, 450), (175, 0, 175), -1)
    cv2.putText(img, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
