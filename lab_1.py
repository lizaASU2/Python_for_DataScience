# Лаб_1.
#1) Используя офисное приложение или текстовый редактор подготовить данные в формате .csv (матрица).
#2) Прочитать эти данные в numpy.
#3) Найти максимум и минимум.  Поменять их местами.

import numpy as np
# создаем массив
x=np.random.randint(100,size = (5,5))
# сохраняем массив в файл
np.savetxt(delimiter= ',', fname='lab_1.csv',fmt='% 1.1i', X=x)
# читаем из файла
y=np.loadtxt(delimiter= ',',fname='lab_1.csv')
print('original matrix: \n',y)
# находим мах и мин элементы 
y_min=np.min(y)
tmp_min=np.where(y==y_min)
y_max=np.max(y)
tmp_max=np.where(y==y_max)
print('max= ',y_max)
#print('position max= ',tmp_max)
print('min= ',y_min)
#print('position min= ',tmp_min)
# меняем местами мах и мин элементы
y[tmp_max]=y_min
y[tmp_min]=y_max
print('new matrix: \n',y)