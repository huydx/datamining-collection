# -*- coding: utf-8 -*-
import os
import scipy as sp
import numpy as np
import math
from scipy.linalg import det, inv
import matplotlib.pyplot as plt

data = []
trajectory = []
normalize_trajectory = np.array([])
X_PROJECT_INDEX = 3
Y_PROJECT_INDEX = 4
HLAC_WINDOWSZ = 5 #高次自己相関関数のウィンダウサイズ
THRESHOLD = 10e-4 #二つの近似ポイントのコサイン尺度誤差は閾値を超えたら特徴ポイント

#高次自己相関関数パータンリスト
PATTERN_LIST = [
  np.array([[0,0,0],[0,1,1],[0,0,0]]),
  np.array([[0,0,1],[0,1,0],[0,0,0]]),
  np.array([[0,1,0],[0,1,0],[0,0,0]]),
  np.array([[1,0,0],[0,1,0],[0,0,0]]),
  np.array([[0,0,0],[1,1,1],[0,0,0]]),
  np.array([[0,0,1],[0,1,0],[1,0,0]]),
  np.array([[0,1,0],[0,1,0],[0,1,0]]),
  np.array([[1,0,0],[0,1,0],[0,0,1]]),
  np.array([[0,0,1],[1,1,0],[0,0,0]]),
  np.array([[0,1,0],[0,1,0],[1,0,0]]),
  np.array([[1,0,0],[0,1,0],[0,1,0]]),
  np.array([[0,0,0],[1,1,0],[0,0,1]]),
  np.array([[0,0,0],[0,1,1],[1,0,0]]),
  np.array([[0,0,1],[0,1,0],[0,1,0]]),
  np.array([[0,1,0],[0,1,0],[0,0,1]]),
  np.array([[1,0,0],[0,1,1],[0,0,0]]),
  np.array([[0,1,0],[1,1,0],[0,0,0]]),
  np.array([[1,0,0],[0,1,0],[1,0,0]]),
  np.array([[0,0,0],[1,1,0],[0,1,0]]),
  np.array([[0,0,0],[0,1,0],[1,0,1]]),
  np.array([[0,0,0],[0,1,1],[0,1,0]]),
  np.array([[0,0,1],[0,1,0],[0,0,1]]),
  np.array([[0,1,0],[0,1,1],[0,0,0]]),
  np.array([[1,0,1],[0,1,0],[0,0,0]])
]

def _unit_test():
  print type(normalize_trajectory)
  print normalize_trajectory.shape[0]
  print normalize_trajectory[1]
  print type(normalize_trajectory[1])
  print normalize_trajectory[1][0]
  print type(normalize_trajectory[1][0])

def calc_relative_coordinate(difx, dify):
  x, y = 0, 0 
  if difx == 0:
    if dify > 0:
      x, y = 0, 1
    if dify < 0:
      x, y = 0,-1
    if dify == 0:
      x, y = 0, 0
  if difx > 0:
    if dify > 0:
      x, y = 1, 1
    if dify < 0:
      x, y = 1, -1
    if dify == 0:
      x, y = 1, 0
  if difx < 0:
    if dify > 0:
      x, y = -1, 1
    if dify < 0:
      x, y = -1, -1
    if dify == 0:
      x, y = -1, 0
  return [x, y]

def make_window(point_idx):
  tr = normalize_trajectory
  nx, ny, px, py = 0, 0, 0, 0
   
  p = tr[point_idx]
  pprev = tr[point_idx-1] if point_idx >= 1 else p
  pnext = tr[point_idx+1] if point_idx <= tr.shape[0] else p
 
  difprev_X = p[0] - pprev[0]
  difprev_Y = p[1] - pprev[1]
  difnext_X = pnext[0] - p[0]
  difnext_Y = pnext[1] - p[1]
  
  temp1 = calc_relative_coordinate(difprev_X, difprev_Y)
  temp2 = calc_relative_coordinate(difnext_X, difnext_Y)
  
  px, py = temp1[0], temp1[1]
  nx, ny = temp2[0], temp2[1]
  
  ret = np.zeros((3,3))
  ret[px,py] = (difprev_X**2 + difprev_Y**2)**0.5 
  ret[nx,ny] = (difnext_X**2 + difnext_Y**2)**0.5

  return ret

def calc_hlac(point_idx):
  tr = normalize_trajectory
  numpat = len(PATTERN_LIST)
  sum = [0]*numpat

  for i in range(numpat): ##i ~ 高次自己相関関数[i]
    pat = PATTERN_LIST[i]
    start = 0 if point_idx - HLAC_WINDOWSZ/2 < 0 else point_idx - HLAC_WINDOWSZ/2
    end = tr.shape[0]-1 if point_idx + HLAC_WINDOWSZ/2 >= tr.shape[0] else point_idx + HLAC_WINDOWSZ/2
    
    ##ウィンダウンの中で高次自己相関関数を計算
    for m in range(start, end+1):
      wd = make_window(m)
      for x in range(3):
        for y in range(3):
          sum[i] = sum[i] + wd[x][y] * pat[x][y] 

  return sum

def hlac_diff(pt1, pt2):
  hl1 = np.asarray(calc_hlac(pt1))
  hl2 = np.asarray(calc_hlac(pt2))
  #コサイン尺度で計算
  return np.dot(hl1, hl2) / (math.sqrt(np.dot(hl1, hl1)) * math.sqrt(np.dot(hl2, hl2)))

def input(file_name, delimiter='|'):
  f = open(file_name)
  f_cont = f.readlines()
  for line in f_cont:
    if len(line) < 1: continue
    _tmp = line.rsplit(delimiter)
    data.append(_tmp)
    trajectory.append([float(_tmp[X_PROJECT_INDEX].rstrip()), float(_tmp[Y_PROJECT_INDEX].rstrip())])

def normalize():
  traject = sp.asarray(trajectory)
  mean = np.mean(traject, axis=0)
  return sp.apply_along_axis(lambda x: x-mean, 1, traject) #正規化

def find_feature_point():
  ret = []
  tr = normalize_trajectory
  ret.append(0)
  for index in range(tr.shape[0]):
    try:
      if index > 0:
        dif = hlac_diff(index-1, index)
        print dif
        if dif < THRESHOLD: ret.append(index)
    except:
      print index
  
  return ret

if __name__ == '__main__':
  input('gpsdata01.txt')
  normalize_trajectory = normalize()
  find_feature_point()
  #_unit_test()

