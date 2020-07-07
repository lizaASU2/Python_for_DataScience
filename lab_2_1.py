# Заача: обучить формальный нейрон научить распозновать гласные и согласные буквы ( буквы представлены как картинки)
# 1 - гласная, 2 - согласная

import numpy as np
import matplotlib.pyplot as plt # работает с изображениями
from scipy import ndimage
import os

y0 = np.array([1,0,1,0,0], dtype=int)
D = []

# считываем изображения из папки
for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img'): 
    pic = np.dot(plt.imread(i)[...,:3],[1,1,1]).astype(int).flatten() # приводим в ч/б, преобразуем в int и переводим в вектор
    D+=[pic]

w = np.zeros(25) # веса

#константы обучения (методом подбора)
α = 0.2 # темп
β = -0.4 # коэф.торможения
σ = lambda x: 1 if x > 0 else 0 #активационная функция
 
# функция распознования
def f(x):
    s = β + np.sum(x @ w)
    return σ(s)

# обучние
def train():
    global w
    _w = w.copy()  # копируем веса, чтобы можно было менять
    # полный перебор
    for x, y in zip(D, y0):
        w += α * (y - f(x)) * x
    return (w != _w).any() #условие прекращения обучения

# обучаем до тех пор, пока не пройдет вся выборка           
while train():
    print(w)

# тест на исходных изображениях
#print('\n Start dateset: \n')
#for i in range(5):
   # print(f(D[i]))
    
# тест на новых изображениях
print('\n New dateset: \n')
for i in os.listdir(r'C:\Users\Лиза\Desktop\Python for Data Science\lab_2\img1'):
    pic1 = np.dot(plt.imread(i)[...,:3],[1,1,1]).astype(int).flatten()
    print(f(pic1))
