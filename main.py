import cv2
import pickle
import numpy as np
import cvzone as cvzone
width, height = 107, 48

try:
    with open('D:\\Study\\project end to end\\Car_parcking_system\\dataset\\CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:  # Corrected event type for right-click
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)

    with open('D:\\Study\\project end to end\\Car_parcking_system\\dataset\\CarParkPos', 'wb') as f:
        pickle.dump(posList, f)


while True:
    img = cv2.imread('D:\\Study\\project end to end\\Car_parcking_system\\dataset\\carParkImg.png')
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    if cv2.waitKey(1) == 27:  # Press Esc key to exit
        break
cv2.destroyAllWindows()



 
cap = cv2.VideoCapture('D:\\Study\\project end to end\\Car_parcking_system\\dataset\\carPark.mp4')
 
with open('D:\\Study\\project end to end\\Car_parcking_system\\dataset\\CarParkPos', 'rb') as f:
    posList = pickle.load(f)
width, height = 107, 48
 
def check_parking_space(img_pro):
    space_counter = 0
 
    for pos in posList:
        x, y = pos
 
        img_crop = img_pro[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)
 
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            space_counter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
 
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)
 
    cvzone.putTextRect(img, f'Free: {space_counter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0, 200, 0))
 
 
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
    success, img = cap.read()
 
    if not success:
        break
 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 16)
    img_median = cv2.medianBlur(img_threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_median, kernel, iterations=1)
 
    check_parking_space(img_dilate)
 
    cv2.imshow("Image", img)
 
    if cv2.waitKey(1) == 27:  # Press Esc key to exit
        break
cv2.destroyAllWindows()