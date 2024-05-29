import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def norm(age, M=57, m=13):
    return (age-13)/(57-13)

def csvfil():
    files = os.listdir('./UTKFace')
    d = {}  # Initialize the dictionary to store file paths and ages

    for file in files:
        age = int(file.split('_')[0])  # Convert age to integer
        if 13 <= age <= 57 and age != 26:
            d['UTKFace/' + file] = norm(age)

    df = pd.DataFrame(list(d.items()), columns=['file_path', 'age'])

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the datasets to CSV files
    train_df.to_csv("train_data.csv", index=False)
    val_df.to_csv("val_data.csv", index=False)

csvfil()
