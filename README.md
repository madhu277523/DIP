#DIP
### Develop a program to display greyscale image using read and write operations.
Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

Importance of grayscaling –

Dimension reduction: For e.g. In RGB images there are three color channels and has three dimensions while grayscaled images are single dimensional.
Reduces model complexity: Consider training neural article on RGB images of 10x10x3 pixel.The input layer will have 300 input nodes. On the other hand, the same neural network will need only 100 input node for grayscaled images.
For other algorithms to work: There are many algorithms that are customized to work only on grayscaled images e.g. Canny edge detection function pre-implemented in OpenCV library works on Grayscaled images only.

## Program:
import cv2
image=cv2.imread('pic.jpg')
cv2.imshow('orinal image',image)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale',gray_image)
cv2.waitKey(0)
cv2.imwrite("rose.jpg',gray_image)
cv2.destroyAllWindows()

## imread()
Loads an image from the specified file.

## imwrite()
THis method is used to save an image to storage devices.

## waitkey()
It is keyboard binding function .The function waits for specified milisecond for any keyboard event.

## destroyAllWindow()
 simply destroy all the windows we created.

![image](https://user-images.githubusercontent.com/72431161/104423517-0e819b00-5533-11eb-8493-a9cadcaa5f9f.png)

## 2.Develop a program to perform linear transformation on a image(Scaling & Rotation).
## a)Scaling
## b) Rotation

## a) Scaling
Scaling is just resizing of the image. OpenCV comes with a function cv.resize() for this purpose. The size of the image can be specified manually, or you can specify the scaling factor.
## program:
import cv2
import numpy as np
src=cv2.imread('pic.jpg')
img=cv2.imshow('pic',src)
cv2.waitKey(0)
scale_p=200
width=int(src.shape[1]*scale_p/100)
height=int(src.shape[0]*scale_p/100)
dsize=(width,height)
result=cv2.resize(src,dsize)
cv2.imshow('scaling',result)
cv2.waitKey(0)
## Output:
![image](https://user-images.githubusercontent.com/72431161/104424529-5c4ad300-5534-11eb-9cb9-a0cda4d1b767.png)
## b)Rotation
## PROGRAM:
import cv2
import numpy as np
src=cv2.imread('pic.jpg')
img=cv2.imshow('pic',src)
windowsname=img
img=cv2.rotate(src,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow(windowsname,img)
cv2.waitKey(0)
Output:
![image](https://user-images.githubusercontent.com/72431161/104425348-6caf7d80-5535-11eb-8568-24656759f92a.png)

## 3. Develop a program to find sum and mean of a set of images.
## Create n number of images and read the directory and perform operation.
## program:
import cv2
import os
path = &#39;C:\Pictures&#39;
imgs = []
files = os.listdir(path)
for file in files:
filepath=path+&quot;\\&quot;+file
imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:

#cv2.imshow(files[i],imgs[i])
im+=imgs[i]
i=i+1
cv2.imshow(&quot;sum of four pictures&quot;,im)
meanImg = im/len(files)
cv2.imshow(&quot;mean of four pictures&quot;,meanImg)
cv2.waitKey(0)
 ## Output:
 ![image](https://user-images.githubusercontent.com/72431161/104425995-33c3d880-5536-11eb-89f7-9729c2054f2c.png)
 
 ## 4.Develop a program to convert a  color image to gray scale and binary image.
import cv2
originalimg=cv2.imread('pic.jpg')
gray_image=cv2.cvtColor(originalimg,cv2.COLOR_BGR2GRAY)
(thresh,blackAndWhiteImage)=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
cv2.imshow('black and whiteimage',blackAndWhiteImage)
cv2.imshow('Original Image',originalimg)
cv2.imshow('Gray Image',gray_image)
cv2.waitKey(0)
## Output:
![image](https://user-images.githubusercontent.com/72431161/104426399-be0c3c80-5536-11eb-9ae0-d22e61910b8e.png)
 ![image](https://user-images.githubusercontent.com/72431161/104426525-e7c56380-5536-11eb-926d-f499474ec6df.png)
 ## 5.Develop a program to convert a color image to different color space.
 ## program:
import cv2 
img = cv2.imread('download1.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
cv2.imshow('hsv', img) 
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
cv2.imshow('lab', img) 
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
cv2.imshow('crcy', img) 
cv2.waitKey(0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('gray', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()

##Output:
![image](https://user-images.githubusercontent.com/72431161/104426746-2fe48600-5537-11eb-909e-6beaa275a6ba.png)
