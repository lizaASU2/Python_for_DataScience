# Заача: оценить качество распознования при повроте изображения

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
 
D=[]

# cоздаем и читаем изображение с буквами и переводим в ч/б   
for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img'): 
    pic = np.dot(plt.imread(i)[...,:3],[1,1,1]).astype(int).flatten() # приводим в ч/б, преобразуем в int и переводим в вектор 
    #y0=(np.array(np.binary_repr(ord(i.split('.')[0]))[2:].zfill(3))).flatten()
    y0=np.binary_repr(ord(i.split('.')[0]))
    y0 = ' '.join(y0)
    y0=np.array(y0.split(),dtype=int)
    D+=[[pic,y0]]
   
w = np.zeros((D[0][0].shape[0], D[0][1].shape[0])) # матрица весов (размер должен быть совместим с вход.данными)

β = -0.4 # коэф.торможения
α = 0.2 # темп

#активационная функция
σ = lambda x: (x > 1).astype(int) #превращение в логическую переменную
 
# функция распознования
def f(x):
    s = β + x @ w
    return σ(s)

# обучние 
def train():
    global w
    _w = w.copy() # копируем веса, чтобы можно было менять
    for x, y in D:
        i = np.where(x>0)
        w[i] += α * (y - f(x))
    return (w != _w).any() #условие прекращения обучения

# обучаем до тех пор, пока не пройдет вся выборка            
while train():
    print(w)

# проверка на исходных данных 
print('\n Reference matrix: \n')
for i in range(5):
    Me=f(D[i][0]) # эталонная матрица
    print(Me)

print('\n Recognition quality: \n') 
# поворачиваем изображение в диапозоне от 0 до 180 градусов с шагом в 30 градусов
M=[]
for k in range(0,180,30):
    for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img'):
        #pic1 = np.dot(plt.imread(i)[...,:3],[1,1,1]).astype(int).flatten()
        rotated = ndimage.rotate(np.dot(plt.imread(i)[...,:3],[1,1,1]), 120, reshape=0)
        r = (rotated > 1).astype(int).flatten()
        M += [r]    
        #print(M)
    print('\n Rotation angle: ',k)
    #print('\n Rotated matrix: \n')    
    #for i in range(5):
        #print(f(M[i]))
    
    # оценка качества распознования
    tr=0
    fls=0
    for j in range(5):
        if (f(M[j])==f(D[j][0])).all():
           tr+=1
        else:
           fls+=1
    print("\n Сorrect recognition: ", tr/(tr+fls)*100)
    print("\n Incorrect recognition: ", fls/(tr+fls)*100,'\n')
        
# функция поворота изображения из примера 
#rez=np.zeros((10,8,8))
#M1=[]  
#def test(k):
    #global M1
    #for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img'):
        #rot = ndimage.rotate(np.dot(plt.imread(i)[...,:3],[1,1,1]), k, reshape=0)
        #r1 = (rot > 1).astype(int).flatten()
       # M1+= [r1]

# вводим случайность, т.к. статистика не работает с неслучайными величинами
#for k in range(90): # поворот на 1 градус в диапозоне от 0 до 90
    #rez[k // 10]+=test(k)