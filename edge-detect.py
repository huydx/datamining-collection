import cv
import argparse

parser = argparse.ArgumentParser(description='edge detect')
parser.add_argument('im',  help='image name')

args = parser.parse_args()

image = cv.LoadImage(args.im, cv.CV_LOAD_IMAGE_GRAYSCALE)
draw_image = cv.LoadImage(args.im)

cornerMap = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
cv.CornerHarris(image,cornerMap,3)

for y in range(0, image.height):
  for x in range(0, image.width):
    harris = cv.Get2D(cornerMap, y, x)
    if harris[0] > 10e-6:
      cv.Circle(draw_image, (x,y), 2, cv.RGB(155, 0, 25))

cv.NamedWindow('w')
cv.ShowImage('w', draw_image) 
cv.WaitKey()
      

