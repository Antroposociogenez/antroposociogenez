#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Генерация случайной фразы на основе новостей из ВК


# In[2]:


#Перед началом установите pip и vk_api
#!pip install vk_api

import vk_api
import csv
import pandas as pd
import re

login = '+79785027356'
passwd = 'White12405'

# Функция поиска новостей в ВК
news = []
def get_data(posts_count, posts):
  for i in range(0, posts_count):
    text = posts['items'][i]['text']
    news.append(text)

# Словарь
names1 = ['tip', 'slova']
df1 = pd.read_csv('/content/drive/MyDrive/kr_slova.csv', sep=',', names=names1)
df1 = df1[~df1['tip'].str.contains('Тип защищаемой информации', na=False)]
strkr = ''
for i in df1['slova']:
  strkr += i + ' '
a = strkr.split()

# Выбор случайного слова из словаря
import random
my_choice = random.choice(a) + '\n'
my_choice

# Получение текстов новостей
vk_session = vk_api.VkApi(login, passwd)
vk_session.auth()
vk = vk_session.get_api()
request = my_choice
posts_count = 1
next_from = 0
if posts_count <= 200:
  posts = vk.newsfeed.search(q = request, count = posts_count, startfrom = next_from)
  get_data(posts_count, posts)
else:
  req_times = posts_count // 200
  last_req = posts_count % 200
  for k in range(0, req_times):
    posts = vk.newsfeed.search(q = request, count = 200, startfrom = next_from)
    next_from = posts['next_from']
    get_data(200, posts)
  posts = vk.newsfeed.search(q = request, count = last_req, startfrom = next_from)
  get_data(last_req, posts)

sen = news[0]
sen = sen.lower()
sen = re.sub(r'[^а-яА-Я\s]', ' ', sen)

w = ''
qw = []
spl = sen.split()
predlogi = ['в', 'без', 'до', 'из', 'к', 'на', 'по', 'о', 'от', 'перед', 'при', 'через', 'с', 'у', 'за', 'над', 'об', 'под', 'про', 'для']
i = 0
while i <= len(spl)-5:
  if spl[i+4] not in predlogi:
    w += spl[i] + ' ' + spl[i+1] + ' ' + spl[i+2] + ' ' + spl[i+3] + ' ' + spl[i+4]
    qw.append(w)
    w = ''
    i += 5
  else:
    i += 1

import random
my_choice = random.choice(qw)
print("Произнесите фразу: ", my_choice)


# In[ ]:


#Распознавание фразы, сказанной в микрофон, и проверка соответствия


# In[ ]:


# ЭТОТ ФРАГМЕНТ РАБОТАЕТ ТОЛЬКО В АНАКОНДЕ, НЕ ЗАПУСКАТЬ ЕГО В КОЛАБ!
# Установите пожалуйста библиотеку SpeechRecognition
# pip install SpeechRecognition
import speech_recognition as sr

def record_volume():
    r = sr.Recognizer()
    with sr.Microphone(device_index = 1) as source:
        print('Настраиваюсь.')
        r.adjust_for_ambient_noise(source, duration=0.5) #настройка посторонних шумов
        print('Слушаю...')
        audio = r.listen(source)
        with open('microphone-results.wav', 'wb') as f:
          f.write(audio.get_wav_data())
    print('Услышала.')
    try:
        query = r.recognize_google(audio, language = 'ru-RU')
        text = query.lower()
        print(f'Вы сказали: {query.lower()}')
        f.close()
    except:
        print('Error')
    return f, query
record_volume()

my_choice = 'капитальный ремонт корпусов и помещений'
auth_rasst = 0
for i in range(len(my_choice)):
  if my_choice[i] == text[i]:
    auth_rasst += 1
sovpad = (auth_rasst / len(my_choice)) * 100
print(str(sovpad) + '%')
if sovpad >= 80:
  print("Аутентификация произведена успешно.")


# In[ ]:


#Распознавание пользователя по голосу


# In[ ]:


import math
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')

# Извлечение признаков из спектрограммы и запись в csv файл
n = len(os.listdir(f'/content/drive/MyDrive/woices'))
dan = []
dn = ['имя', 'среднеквадратичное отклонение', 'временное преобразование Фурье', 'спектральный центроид', 'спектральная ширина', 'спад спектра', 'частота пересечения нуля', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'MFCC14', 'MFCC15', 'MFCC16', 'MFCC17', 'MFCC18', 'MFCC19', 'MFCC20']
for filename in os.listdir(f'/content/drive/MyDrive/woices'):
  woicename = f'/content/drive/MyDrive/woices/{filename}'
  y, sr = librosa.load(woicename, mono=True, duration=30)
  rmse = librosa.feature.rms(y=y)
  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
  spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
  spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
  zcr = librosa.feature.zero_crossing_rate(y)
  mfcc = librosa.feature.mfcc(y=y, sr=sr)
  d1 = [filename, np.mean(rmse), np.mean(chroma_stft), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
  for i in mfcc:
    d1.append(np.mean(i))
  dan.append(d1)
features = pd.DataFrame(dan, columns=dn)
features.to_csv('features.csv', index=False, sep=';')

names = ['Кирилл', 'Котик', 'Никита1', 'Никита2', 'Jane', 'Dima1', 'Dima', 'Ilea', 'Vera', 'Julia']
pr = ['среднеквадратичное отклонение', 'временное преобразование Фурье', 'спектральный центроид', 'спектральная ширина', 'спад спектра', 'частота пересечения нуля', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'MFCC14', 'MFCC15', 'MFCC16', 'MFCC17', 'MFCC18', 'MFCC19', 'MFCC20']
simils = []

for j in range(10):
  x = 0
  for i in pr:
    x += df1[i][j]/max(df1[i]) * df1[i][10]/max(df1[i])
  ya = 0
  yb = 0
  for i in pr:
    ya += (df1[i][j]/max(df1[i])) ** 2
    yb += (df1[i][10]/max(df1[i])) ** 2
  y = (ya ** (1/2)) * (yb ** (1/2))
  simil = x / y
  simils.append(simil)
sim = pd.DataFrame([simils], columns=names)
sim.to_csv('sim.csv', index=False, sep=';')

maxs = max(simils)
print('Вероятно, голос принадлежит пользователю с именем ',names[simils.index(maxs)])


# In[ ]:


#Голосовое управление банкоматом


# In[ ]:


# ЭТОТ ФРАГМЕНТ РАБОТАЕТ ТОЛЬКО В АНАКОНДЕ, НЕ ЗАПУСКАТЬ ЕГО В КОЛАБ!

print('Выберите функцию и произнесите её название:')
print('Показать счёт')
print('Перевести деньги')
print('Пополнить счёт')
print('Вывести наличные')
def record_volume():
    r = sr.Recognizer()
    with sr.Microphone(device_index = 1) as source:
        print('Настраиваюсь.')
        r.adjust_for_ambient_noise(source, duration=0.5) #настройка посторонних шумов
        print('Слушаю...')
        audio = r.listen(source)
    print('Услышала.')
    try:
        query = r.recognize_google(audio, language = 'ru-RU')
        text = query.lower()
        print(f'Вы сказали: {query.lower()}')
        f.close()
    except:
        print('Error')
record_volume()
if text == 'показать счёт':
  print('На Вашем счету *** рублей.')
elif text == 'перевести деньги':
  print('Скажите, какую сумму Вы хотите перевести?')
  record_volume()
  sum = re.sub(r'[^0-9]', '', text)
  num = input('Введите номер карты получателя.')
  print('Перевод успешно завершён.')
elif text == 'пополнить счёт':
  print('Введите купюру в банкомат.')
elif text == 'вывести наличные':
  print('Скажите, какую сумму Вы хотите вывести?')
  record_volume()
  sum = re.sub(r'[^0-9]', '', text)
else:
  print('Команда введена некорректно.')

