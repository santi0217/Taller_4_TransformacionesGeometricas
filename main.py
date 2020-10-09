#
#   Fecha: 8 de octubre del 2020
#
#   Autores: Santiago Márquez Álvarez
#           Sergio Mora Pradilla
#
#   Descripción: En este script se pretende comparar las diferencias, entre transformación de afinidad y
#               transformación de similitud,  para analizar como se ve afectada a  una imagen en cuanto a
#               translación, rotación y escalado.
#

# Importación de librerias

import numpy as np
import math
import cv2
#iniciación de variables
corx = [0, 0, 0]
cory = [0, 0, 0]
cont = 0
#funcion para detectar click
def click_event(event, x, y, flags, params):
    # cuando se de click izquierdo se guardan las coordenas y se ponen en la imagen
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        global cx,cy, cont
        corx[cont]=x
        cory[cont]=y
        cont=cont+1
        if(cont<4):                                                                 #para no dar mas de 3 puntos
            font = cv2.FONT_HERSHEY_SIMPLEX                                         #tipo de fuente
            cv2.putText(img,str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)  #Coord. X y Y
            cv2.imshow('image', img)
        if(cont>=3):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,str("OPRIMA ENTER !!!"), (0,30 ), font, 1, (255, 0, 0), 2)#aviso para continuar
            cv2.imshow('image', img)

if __name__ == "__main__":
    #obtener direccion de archivo
    path1=input("inserte la direccion de la imagen 1 con el nombre y extension al final")
    image1 = cv2.imread(path1)
    img = image1.copy()
    # mostrar imagen
    cv2.imshow('image', img)
    #obtener coordenadas
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    I1_coor_X=corx.copy()
    I1_coor_Y=cory.copy()
    cont=0

    # obtener direccion de archivo
    path2 =input("inserte la direccion  de la imagen 2 con el nombre y extension al final")
    image2 = cv2.imread(path2)
    img = image2.copy()
    # mostrar imagen
    cv2.imshow('image', img)
    # obtener coordenadas
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    I2_coor_X=corx.copy()
    I2_coor_Y=cory.copy()

    #muestra corrdenadas
    print("cordenadas imagen1 \nEn x", I1_coor_X)
    print("En y", I1_coor_Y)
    print("cordenadas imagen2 \nEn x", I2_coor_X)
    print("En y", I2_coor_Y)

    #lista de puntos imagen 1
    pts1 = np.float32([[I1_coor_X[0], I1_coor_Y[0]], [I1_coor_X[1], I1_coor_Y[1]], [I1_coor_X[2], I1_coor_Y[2]]])
    #lista de puntos imagen 2
    pts2 = np.float32([[I2_coor_X[0], I2_coor_Y[0]], [I2_coor_X[1], I2_coor_Y[1]], [I2_coor_X[2], I2_coor_Y[2]]])

    #matriz de afinidad
    M_affine = cv2.getAffineTransform(pts1, pts2)
    print(M_affine)
    #imagen de afinidad
    image_affine = cv2.warpAffine(image2, M_affine, image1.shape[:2])

    H_matrix=M_affine.copy()

    #usando metodos algebraicos se obtiene los valores de de escalado *REF
    s0=math.sqrt(  (H_matrix[0,0]**2) +(H_matrix[1,0]**2) )
    s1=math.sqrt(  (H_matrix[0,1]**2) +(H_matrix[1,1]**2) )

    # usando metodos algebraicos se obtiene los valores del angulo *REF
    tetha=((1)*np.arctan(H_matrix[1,0]/H_matrix[0,0] ))
    # usando metodos algebraicos se obtiene los valores del angulo *REF
    tx=(H_matrix[0,2]*np.cos(tetha)-H_matrix[1,2]*np.sin(tetha))/s0
    ty=(H_matrix[0,2]*np.sin(tetha)-H_matrix[1,2]*np.cos(tetha))/s1

    print(s0,s1,tetha,tx,ty)
    cv2.imshow("Image normal", image_affine)
    cv2.waitKey(0)

    #matriz  de militud con la imagen 1
    M_sim = np.float32([[s0 * np.cos(tetha), -np.sin(tetha), tx],[np.sin(tetha), s1 * np.cos(tetha), ty]])
    image_similarity = cv2.warpAffine(image1, M_sim, image1.shape[:2])
    cv2.imshow("Image warpped", image_similarity)
    cv2.waitKey(0)

    #REF: 9_GeometricTransformations.pdf, lecture 2: Geometric image transformation, H. rhody, september 9, 2005 pg.13-26
