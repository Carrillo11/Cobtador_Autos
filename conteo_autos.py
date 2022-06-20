from turtle import width
import cv2
import numpy as np
import imutils

video = cv2.VideoCapture('autos.mp4') #Importamos el video a analizar

#Declaracion de algoritmo se sustraccion de fondo.
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

#Mejora la imagen binaria, luego de utilizar la sustraccion de fondo. 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

#Contador que inicia con 0
car_counter = 0 

while True:

    ret, frame =  video.read()
    if ret == False: break
    frame = imutils.resize(frame, width = 640)

    #Se delimitan los puntos extremos de el area que se desea analizar.
    area_pts = np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330,271]])

    #Por medio una una imagen auxiliar, se determinara el area de conteo de autos.
    imAux = np.zeros(shape=(frame.shape[:2]), dtype = np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask = imAux)

    #Se aplica sustraccion de fondo.
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations = 5)

    #Se encuentran los contornos presentes en fgmask, en funcion de su area se puede determinar movimiento de autos.
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

            #Si el auto se encuentra entre 440 y 460 entonces se incrementa en 1 el contador
            if 440 < (x + w) < 460:
                car_counter = car_counter + 1
                cv2.line(frame, (450, 216), (450, 271), (0, 255, 0), 3)

    #Visualizacion del area delimitada.
    cv2.drawContours(frame, [area_pts], -1, (255,0,255), 2)
    cv2.line(frame, (450,216), (450, 271), (0, 255, 255), 1)
    #Permite crear un rectangulo en el cual se tendra el dato de la cantidad de autos.
    cv2.rectangle(frame, (frame.shape[1]-70, 215), (frame.shape[1]-5, 270), (0, 255, 0), 2)
    #El valor se incrementa cuando un auto pasa por la linea amarilla.
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55,250),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow('Frame', frame)
    #cv2.imshow('fgmask', fgmask)

    k = cv2.waitKey(70) & 0xFF
    if k == 27: break

video.release()
cv2.destroyAllWindows()