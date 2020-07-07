# Заача: переделать формальный нейрон в персептрон и научить распозновать гласные и согласные буквы ( буквы представлены как картинки)
# 1 - гласная, 2 - согласная

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
print('\n Start dateset: \n')
for i in range(5):
    print(f(D[i][0]))

# тест на новых изображениях
print('\n New dateset: \n')
for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img1'):
    pic1 = np.dot(plt.imread(i)[...,:3],[1,1,1]).astype(int).flatten()
    print(f(pic1))
