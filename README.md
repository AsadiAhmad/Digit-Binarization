# Digit-Binarization
Binarization Digits of numbers and prepare digits for OCR or number detection and remove the noise and shadow (background) from the picture completely.

<div display=flex align=center>
    <img src="/Pictures/0.jpg" width="400px"/>
    <img src="/Pictures/5.jpg" width="400px"/>
</div>


## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python

## Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Digit-Binarization/blob/main/Code/Image_Binarization.ipynb)

## Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `cv2_imshow`

```sh
import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np
```

### Step 2: Download Image

We need to Download the images from my `Github` repository or you can download others that have digits by your own.

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Digit-Binarization/main/Pictures/number0.jpg -O number0.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Digit-Binarization/main/Pictures/number1.jpg -O number1.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Digit-Binarization/main/Pictures/number2.jpg -O number2.jpg
```

### Step 3: Load Images

we need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`

```sh
image = cv.imread('number0.jpg', cv.IMREAD_GRAYSCALE)
```

<div display=flex align=center>
  <img src="/Pictures/0.jpg" width="400px"/>
</div>

### Step 4: Image Binarization

this is our primary state that we should remove noise with median filter and then use the adaptive thresholding for removing the background in each section of the image so we do not have any dark section in the image.

```sh
noise_removed = cv.medianBlur(image, 5)
binary_image = cv.adaptiveThreshold(noise_removed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
```

<div display=flex align=center>
  <img src="/Pictures/1.jpg" width="400px"/>
</div>

### Step 5: Invert the Binary Image

For using the morpholgy in image processing we need to invert the images 

```sh
inverted_image = 255 - binary_image
```

<div display=flex align=center>
  <img src="/Pictures/2.jpg" width="400px"/>
</div>

### Step 6: Opening Image for Completely remove Noise

Actually Opening have two section :

1- Erosion for removing noise that are not eleminated by midan filter and created after biniarization the image.

```sh
kernel = np.ones((2, 2), np.uint8)
erosion = cv.erode(inverted_image, kernel, iterations = 1)
```

<div display=flex align=center>
  <img src="/Pictures/3.jpg" width="400"/>
</div>

2- Dilation for bolding the text because after the erosion we lose some part of the text so we need to refill the text.

```sh
kernel2 = np.ones((5, 5), np.uint8)
dilation = cv.dilate(erosion, kernel2, iterations = 1)
```

<div display=flex align=center>
  <img src="/Pictures/4.jpg" width="400px"/>
</div>

### Step 7: Invert Image again

we have an Image with white text and black background and we don't want this so we invert that again.

```sh
inverted_image2 = 255 - dilation
```

<div display=flex align=center>
  <img src="/Pictures/5.jpg" width="400px"/>
</div>

### Step 8: All together for other images

so in this step we put all of things together and test that for other images.

```sh
def binarization_image(image, blur_value=5, kernel_erosion=(2, 2), kernel_dilation=(5, 5)):
    noise_removed = cv.medianBlur(image, blur_value)
    binary_image = cv.adaptiveThreshold(noise_removed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    inverted_image = 255 - binary_image
    kernel = np.ones(kernel_erosion, np.uint8)
    erosion = cv.erode(inverted_image, kernel, iterations = 1)
    kernel2 = np.ones(kernel_dilation, np.uint8)
    dilation = cv.dilate(erosion, kernel2, iterations = 1)
    inverted_image2 = 255 - dilation
    return inverted_image2
```

<div display=flex align=center>
  <img src="/Pictures/number1.jpg" width="400px"/>
</div>

<div display=flex align=center>
  <img src="/Pictures/6.jpg" width="400px"/>
</div>

<div display=flex align=center>
  <img src="/Pictures/number2.jpg" width="400px"/>
</div>

<div display=flex align=center>
  <img src="/Pictures/7.jpg" width="400px"/>
</div>

## Parameters

these are parameters of the digit binarization.

- blur_value : the format is like a number `5` higher value higher remove noise and sometime lose the parts of the main digit
- kernel_erosion : the format is like a tuple `(2, 2)` this is a kernel for erosion higher value of this kernal can remove more noise but maybe we lose some part of the main digit
- kernel_dilation : the format is like a tuple `(5, 5)` this is a kernel for dilation higher value of this kernel bolded the digit more and lower kernel value can make digit unreadable after the erosion!

So for each image can vary depending on the size of the digit or size or shape of the noise.

## License

This project is licensed under the MIT License.
