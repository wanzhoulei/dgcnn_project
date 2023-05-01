import os
import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as T

def get_image_list(path):
    return np.array(os.listdir(path))

def get_image_embedding(image, y_dim, x_dim):
    embed = np.zeros((y_dim*x_dim, 5))
    index = 0
    for y_i in range(y_dim):
        for x_i in range(x_dim):
            embed[index][0] = x_i/x_dim #x position normalized by x_dim
            embed[index][1] = y_i/x_dim #y position 
            embed[index][2] = float(image[:, y_i , x_i][0]) # r value
            embed[index][3] = float(image[:, y_i , x_i][1]) # g value
            embed[index][4] = float(image[:, y_i , x_i][2]) # b value
            index += 1
    return embed

def get_label_embedding(label):
    return label[0].flatten().numpy()

##this function reads the image by the link and then resize and split the image into four subimages
##it returns four tuples (img_embed, label_embed) of the four subimages
##img_embed is numpy.ndarray of shape (101*150, 5) and label_embed is 1d numpy array
def read_one_image(img_path, rootpath, image_label_dict, resize_size = (300, 202)):
    img = Image.open(rootpath+"/{}".format(img_path))
    img_label = Image.open(rootpath+"/{}".format(image_label_dict[img_path]))
    ##convert to tensor
    img_tensor = ToTensor()(img)
    label_tensor = ToTensor()(img_label)
    ##resize the original PIL image
    resize_img = img.resize(resize_size)
    resize_label = img_label.resize(resize_size)
    ##convert the resized pil image into tensor
    resize_img_tensor = ToTensor()(resize_img)
    resize_label_tensor = (ToTensor()(resize_label)>0)*1
    ##split the image into four parts
    y_len = int(resize_size[1]/2)
    x_len = int(resize_size[0]/2)
    upper_left_img = resize_img_tensor[:,:y_len, :x_len]
    upper_right_img = resize_img_tensor[:, :y_len, x_len:]
    lower_left_img = resize_img_tensor[:, y_len:, :x_len]
    lower_right_img = resize_img_tensor[:, y_len:, x_len:]
    ##split the label into four parts
    upper_left_label = resize_label_tensor[:,:y_len, :x_len]
    upper_right_label = resize_label_tensor[:, :y_len, x_len:]
    lower_left_label = resize_label_tensor[:, y_len:, :x_len]
    lower_right_label = resize_label_tensor[:, y_len:, x_len:]
    y_dim = int(resize_size[1]/2); x_dim = int(resize_size[0]/2)
    ##compute the embedding for the images
    upper_left_img_embed = get_image_embedding(upper_left_img, y_dim, x_dim)
    upper_right_img_embed = get_image_embedding(upper_right_img, y_dim, x_dim)
    lower_left_img_embed = get_image_embedding(lower_left_img, y_dim, x_dim)
    lower_right_img_embed = get_image_embedding(lower_right_img, y_dim, x_dim)
    ##compute the embedding for the labeling
    upper_left_label_embed = get_label_embedding(upper_left_label)
    upper_right_label_embed = get_label_embedding(upper_right_label)
    lower_left_label_embed = get_label_embedding(lower_left_label)
    lower_right_label_embed = get_label_embedding(lower_right_label)
    X = np.array([upper_left_img_embed, upper_right_img_embed, lower_left_img_embed, lower_right_img_embed])
    y = np.array([upper_left_label_embed, upper_right_label_embed, lower_left_label_embed, lower_right_label_embed])
    return X, y

##this function reads all images and transform each of them into four subparts
##returns a dataset as tuple
##it returns a numpy array of tuples 
#each tuple is a image label pair
def load_dataset(image_list, rootpath, diction, resize_size = (300, 202)):
    X = np.array([]); Y = np.array([])
    i=1
    for name in image_list:
        if i%10 == 0:
            print("Processing {}th Image.".format(i))
        i+=1
        x, y = read_one_image(name, rootpath, diction, resize_size = resize_size)
        if len(X) == 0:
            X = x; Y = y
        else:
            X = np.concatenate((X, x), axis = 0)
            Y = np.concatenate((Y, y), axis = 0)
    return X, Y

def load(path, resize=(300, 202)):
    images_list = get_image_list(path)
    labels_image = sorted(images_list[['anno' in x for x in images_list]])
    pixels_image = sorted(images_list[['anno' not in x and '.bmp' in x for x in images_list]])
    ##construct a map that maps image file name to label file name
    image_label_dict = {name: name.split('.')[0] + '_anno.bmp' for name in pixels_image}
    return load_dataset(pixels_image, path, image_label_dict, resize_size=resize)














