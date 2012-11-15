import cv
import sys
import argparse 
import math

parser = argparse.ArgumentParser(description='optical flow')
parser.add_argument('im1',  help='image 1')
parser.add_argument('im2',  help='image 2')
parser.add_argument('algorithm', help='algorithm')

args = parser.parse_args()

src_im1 = cv.LoadImage(args.im1, cv.CV_LOAD_IMAGE_GRAYSCALE)
src_im2 = cv.LoadImage(args.im2, cv.CV_LOAD_IMAGE_GRAYSCALE)

##using horn schunk
if args.algorithm == 'HS':
  dst_im1 = cv.LoadImage(args.im2, cv.CV_LOAD_IMAGE_COLOR)
  dst_im2 = dst_im1

  #size is tuple type
  cols = src_im1.width
  rows = src_im1.height

  velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
  vely = cv.CreateMat(rows, cols, cv.CV_32FC1)

  cv.SetZero(velx)
  cv.SetZero(vely)


  cv.CalcOpticalFlowHS(src_im1, src_im2, 0, velx, vely, 100.0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))
  #cv.CalcOpticalFlowLK(src_im1, src_im2, (10,10), velx, vely)
  print velx
  print vely

  for i in range(0, (cols-1), 5):
    for j in range(0, (rows-1), 5):
      dx = cv.GetReal2D(velx, j, i)
      dy = cv.GetReal2D(vely, j, i)
      cv.Line(dst_im1, (i, j), (int(i+dx), int(j+dy)), cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0)
    

  cv.NamedWindow("w", cv.CV_WINDOW_AUTOSIZE)
  cv.ShowImage("w", dst_im1)
  cv.WaitKey()

##using Lucas Kanade
if args.algorithm == 'LK':
  dst_img = cv.LoadImage(args.im2, cv.CV_LOAD_IMAGE_COLOR)
  eign_img = cv.CreateImage(cv.GetSize(src_im1), cv.IPL_DEPTH_32F, 1)
  temp_img = cv.CreateImage(cv.GetSize(src_im1), cv.IPL_DEPTH_32F, 1)
  features = cv.GoodFeaturesToTrack(src_im1, eign_img, temp_img, 5000,  0.1, 10, None, True)
  #features = []
  #for i in range(1, dst_img.width, 1):
  #  for j in range(1, dst_img.height, 1):
  #    features.append((i,j)) 
  
  r = cv.CalcOpticalFlowPyrLK(src_im1, src_im2, None, None, features, (100,100), 0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01) ,0)
  list = r[0]
  for i in range(len(list)) :
    dis = math.sqrt(math.pow((features[i][0]-list[i][0]),2) + math.pow((features[i][1]-list[i][1]),2))
    cv.Line(dst_img, (int(features[i][0]), int(features[i][1])), (int(list[i][0]), int(list[i][1])), cv.CV_RGB(255, 0, 0), 1, cv.CV_AA, 0)

  cv.NamedWindow("w", cv.CV_WINDOW_AUTOSIZE)
  cv.ShowImage("w", dst_img)
  cv.WaitKey()


