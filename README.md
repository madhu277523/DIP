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

