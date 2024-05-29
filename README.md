# Face-Age-Pred

## About

The aim is to be able to take an image of a person and be able to detect the range of age within 5 yrs error gap. Facial features are recognized by several layers of convolution and at the end produce a regressive value for the age. All the data has been normalised, since they perform better than usual.

## Steps to run the code
- git clone the repo
- Download the dataset UTKFace from the github repo {}
- run main.py on terminal
- training might take 1.5 hrs in a GPU aided machine

## Pointers about the code
- Age prediction is done only for ages between 13 to 57 yrs

## Results:
- Since the data is not distributed uniformly the margin of age predicted is slightly large - 7yrs
- Age for Saina Nehwal in following picture:  
    ![](saina.jpg)  
  - True Age - 26 yrs, Predicted age - 24 yrs

- Age for Sachin Tendular in following picture:  
    ![](sachin.webp)
  - True Age - x yrs, Predicted age - 39 yrs

  
