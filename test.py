import cv2
import numpy as np
import face_recognition
import os

path = '/home/islam/Documents/face/src/persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  
            encodeList.append(encodings[0])
        else:
            print("No face detected in one of the images. Skipping.")
    return encodeList


encodeListKnown = findEncodeings(images)
print('Encoding Complete.')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)
        if faceDis[matchIndex] < 0.5:
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                if name == "MABROUK" :
                    name = "Mabrouk Logbibi \nCSA kernel in OSCA"
                elif name == "KHALIL" :
                    name = "Kalil Chaddadi \nOSCA head"
                elif name == "AHMED" :
                    name = "Ahmed Kassas \nINFOSEC kernel in OSCA"
                elif name == "SOSIA" : 
                    name = "Sofia lalagui \nevent organiser in OSCA"
                
                print(name)
                

                name_lines = name.split("\n")

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                box_height = 35 + (len(name_lines) - 1) * 20  
                padding = 50 
                y_top = y1 - padding  

                if y_top < 0:
                    y_top = 0
                max_text_width = 0
                for line in name_lines:
                    (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
                    max_text_width = max(max_text_width, text_width)
                box_width = max_text_width + 20  

                cv2.rectangle(img, (x1, y_top), (x1 + box_width, y_top + box_height), (77, 0, 153), cv2.FILLED)
                total_text_height = sum([cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0][1] for line in name_lines])
                

                vertical_offset = (box_height - total_text_height) // 2

                for i, line in enumerate(name_lines):
                    text_width, text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
                    x_center = x1 + (box_width - text_width) // 2
                    y_position = y_top + vertical_offset + i * (text_height + 5)  

                    cv2.putText(img, line, (x_center, y_position), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (77, 0, 153), 2)

            cv2.imshow('Face Recognition', img)
            cv2.waitKey(8)
        else:
            print("unknown")

        
