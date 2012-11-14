import cv
import sys
import argparse 

parser = argparse.ArgumentParser(description='optical flow')
parser.add_argument('im1',  help='image 1')
parser.add_argument('im2',  help='image 2')

args = parser.parse_args()

src_im1 = cv.LoadImage(args.im1, cv.CV_LOAD_IMAGE_GRAYSCALE)
src_im2 = cv.LoadImage(args.im2, cv.CV_LOAD_IMAGE_GRAYSCALE)

dst_im1 = cv.LoadImage(args.im2, cv.CV_LOAD_IMAGE_COLOR)
dst_im2 = dst_im1

#size is tuple type
##todo compare sz1 and sz2
cols = src_im1.width
rows = src_im1.height

velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
vely = cv.CreateMat(rows, cols, cv.CV_32FC1)

cv.SetZero(velx)
cv.SetZero(vely)


cv.CalcOpticalFlowHS(src_im1, src_im2, 0, velx, vely, 100.0, (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))

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



