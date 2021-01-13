#DIP
### 1. Develop a program to display greyscale image using read and write operations.
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

## waitKey()
It is keyboard binding function .The function waits for specified milisecond for any keyboard event.

## destroyAllWindows()
 simply destroy all the windows we created.

##Output


![image](https://user-images.githubusercontent.com/72431161/104423517-0e819b00-5533-11eb-8493-a9cadcaa5f9f.png)

## 2.Develop a program to perform linear transformation on a image(Scaling & Rotation).
## a)Scaling
## b) Rotation

## a)Scaling
Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image 

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
## cv2.resize() 
 This method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the number of pixels from an image .
 ## imshow()
 imshow() function in pyplot module of matplotlib library is used to display data as an image

## Output:
![image](https://user-images.githubusercontent.com/72431161/104424529-5c4ad300-5534-11eb-9cb9-a0cda4d1b767.png)
## b) Rotation
Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal, flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing.

## Program:
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
path = 'C:\Pictures'
imgs = []
files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
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
 ## append() 
 This method in python adds a single item to the existing list.
## listdir() 
This method in python is used to get the list of all files and directories in the specified directory.

 ## Output:
 ![image](https://user-images.githubusercontent.com/72431161/104425995-33c3d880-5536-11eb-89f7-9729c2054f2c.png)
 
 ## 4.Develop a program to convert a  color image to gray scale and binary image.
 Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.
A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white.

import cv2
originalimg=cv2.imread('pic.jpg')
gray_image=cv2.cvtColor(originalimg,cv2.COLOR_BGR2GRAY)
(thresh,blackAndWhiteImage)=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
cv2.imshow('black and whiteimage',blackAndWhiteImage)
cv2.imshow('Original Image',originalimg)
cv2.imshow('Gray Image',gray_image)
cv2.waitKey(0)

 cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black). 

## Output:

![image](https://user-images.githubusercontent.com/72431161/104426399-be0c3c80-5536-11eb-9ae0-d22e61910b8e.png)
 ![image](https://user-images.githubusercontent.com/72431161/104426525-e7c56380-5536-11eb-926d-f499474ec6df.png)
 
 ## 5.Develop a program to convert a color image to different color space.
 A color space is actually a combination of two things: a color model and a mapping function.There are many different color spaces that are useful. Some of the more popular color spaces are RGB, YUV, HSV, Lab, and so on.
 ## HSV
 It is often more natural to think about a color in terms of hue and saturation than in terms of additive or substractive color component.
 ## HSL
 Hue is a degree on the color wheel from 0 to 360. saturation is a percentage value.Lightness is also percentage value. 0% represent black and 100%  represent white.
 ## Lab
 L stand for lightnessand a,b for the color spectrums green-red and blue yellow.
 ## YUV
 It is a color encoding system typically used as part of color image pipelines
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

## cv2.cvtColor()
This method is used to convert an image from one color space to another.   

##Output:

![image](https://user-images.githubusercontent.com/72431161/104426746-2fe48600-5537-11eb-909e-6beaa275a6ba.png)

## 6.develop a program to create an image from 2D array. Generate array of random size.
## program:
## 2D array:
Two dimensional array is an array within an array.
It is the type of array,the position of an data element is reffered by two indices instead of one.However, 2D arrays are created to implement a relational database look alike data structure.

## Program
import numpy as np
from PIL import Image
import cv2 as C
array = np.zeros([100,200,3],dtype=np.uint8)
array[:,:100]=[150,128,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('download.jpg')
img.show()
C.waitKey(0)
## np.zeros()
It returns a new array of given shape and type with zeros.

## Output:
![image](https://user-images.githubusercontent.com/72431161/104427105-b4370900-5537-11eb-9fb5-3d20f9ee107b.png)

