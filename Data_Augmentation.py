import tensorflow as tf
import os
from PIL import Image
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 50
aug_num = 1 # number of times that we augment that data with ImageDataGenerator

def augment(class_list):

    # # UNCOMMENT TO COPY DIRECTORY
    copy_dir = "copy_dir" 
    dst_path = os.path.join('../Skin_Lesions', 'copy_dir')
    print(dst_path)
    if os.path.exists(dst_path) and os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(copy_dir)
    src_path = '../Skin_Lesions/base_dir'
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    # for now, we will not enhance the benign lesions
    shutil.rmtree('../Skin_Lesions/copy_dir/train_dir/nv') 
    shutil.rmtree('../Skin_Lesions/copy_dir/val_dir')


    for dx in class_list:
        datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')
        cpy_path = '../Skin_Lesions/copy_dir/train_dir/'
        
        dst_path = os.path.join('../Skin_Lesions/base_dir/train_dir', dx)
        # os.mkdir(dst_path)


        aug_datagen = datagen.flow_from_directory(cpy_path,
                                        save_to_dir=dst_path,
                                        save_format='png',
                                        target_size=(224,224),
                                        batch_size=batch_size)
        
        # total number of images we want to have in each class
        num_images_wanted = 7000 
        num_files = len(os.listdir(dst_path))
        num_batches = int(np.ceil((num_images_wanted-num_files)/batch_size))

        # run the generator and create about 7000 augmented images
        for i in range(0,num_batches):
            imgs, labels = next(aug_datagen)
        # delete temporary directory with the raw image files
        shutil.rmtree(os.path.join('../Skin_Lesions/copy_dir/train_dir', dx))

    # delete copy directory
    shutil.rmtree('../Skin_Lesions/copy_dir')
    

def main():
    

    class_list = ['mel','bkl','bcc','akiec','vasc','df']
    augment(class_list)

    # print("# of images in training directory")
    # print("nv ", len(os.listdir('base_dir/train_dir/nv')))
    # print("mel ", len(os.listdir('base_dir/train_dir/mel')))
    # print("bkl ", len(os.listdir('base_dir/train_dir/bkl')))
    # print("bcc ", len(os.listdir('base_dir/train_dir/bcc')))
    # print("akiec ", len(os.listdir('base_dir/train_dir/akiec')))
    # print("vasc ", len(os.listdir('base_dir/train_dir/vasc')))
    # print("df ", len(os.listdir('base_dir/train_dir/df')))

    # print()
    # print()

    # print("# of images in validation directory")
    # print("nv ", len(os.listdir('base_dir/val_dir/nv')))
    # print("mel ", len(os.listdir('base_dir/val_dir/mel')))
    # print("bkl ", len(os.listdir('base_dir/val_dir/bkl')))
    # print("bcc ", len(os.listdir('base_dir/val_dir/bcc')))
    # print("akiec ", len(os.listdir('base_dir/val_dir/akiec')))
    # print("vasc ", len(os.listdir('base_dir/val_dir/vasc')))
    # print("df ", len(os.listdir('base_dir/val_dir/df')))




if __name__ == "__main__":
    main()