import cv2
import csv
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)


class MCQ:
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None

    def update(self, cursor, bboxs):
        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = x + 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)


# Import csv file data
pathCSV = "C:/Users/rites/Downloads/Mcqs.csv"
with open(pathCSV, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# Create Object for each MCQ
mcqList = [MCQ(q) for q in dataAll]

print("Total MCQ Objects Created:", len(mcqList))

qNo = 0
qTotal = len(dataAll)


def draw_text_rect(image, text, pos, scale, thickness, offset=10, border=2, color=(255, 0, 255), text_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = pos
    top_left = (x - offset, y - text_size[1] - offset)
    bottom_right = (x + text_size[0] + offset, y + offset)
    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(image, text, (x, y), font, scale, text_color, thickness)
    return image, [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if qNo < qTotal:
        mcq = mcqList[qNo]

        img, bbox = draw_text_rect(img, mcq.question, [100, 100], 1, 2)
        img, bbox1 = draw_text_rect(img, mcq.choice1, [100, 250], 1, 2)
        img, bbox2 = draw_text_rect(img, mcq.choice2, [400, 250], 1, 2)
        img, bbox3 = draw_text_rect(img, mcq.choice3, [100, 400], 1, 2)
        img, bbox4 = draw_text_rect(img, mcq.choice4, [400, 400], 1, 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lmList = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]
                cursor = lmList[8]
                length = int(((lmList[8][0] - lmList[12][0])**2 + (lmList[8][1] - lmList[12][1])**2) ** 0.5)
                print(length)
                if length < 35:
                    mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4])
                    if mcq.userAns is not None:
                        time.sleep(0.3)
                        qNo += 1
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        score = sum(1 for mcq in mcqList if mcq.answer == mcq.userAns)
        score = round((score / qTotal) * 100, 2)
        img, _ = draw_text_rect(img, "Quiz Completed", [250, 300], 1, 2)
        img, _ = draw_text_rect(img, f'Your Score: {score}%', [700, 300], 1, 2)

    # Draw Progress Bar
    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (255, 0, 255), 5)
    img, _ = draw_text_rect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 1, 2)

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
