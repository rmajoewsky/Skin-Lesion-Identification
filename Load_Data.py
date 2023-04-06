import os
import shutil
from PIL import Image 
import pandas as pd


def load_data():
    #read in collection of multi-source dermatoscopic images of pigmented lesions
    os.listdir('../Skin_Lesions/archive')

    #create new directories
    base_dir = 'base_dir'
    
    dirpath = os.path.join('../Skin_Lesions', 'base_dir')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(base_dir)

    training_dir = os.path.join(base_dir, 'train_dir')
    val_dir = os.path.join(base_dir, 'val_dir')
    os.mkdir(training_dir)
    os.mkdir(val_dir)

    #Create folders in Training Directory
    nv = os.path.join(training_dir, 'nv')
    os.mkdir(nv)
    mel = os.path.join(training_dir, 'mel')
    os.mkdir(mel)
    bkl = os.path.join(training_dir, 'bkl')
    os.mkdir(bkl)
    bcc = os.path.join(training_dir, 'bcc')
    os.mkdir(bcc)
    akiec = os.path.join(training_dir, 'akiec')
    os.mkdir(akiec)
    vasc = os.path.join(training_dir, 'vasc')
    os.mkdir(vasc)
    df = os.path.join(training_dir, 'df')
    os.mkdir(df)

    # Create folders in validation directory
    nv = os.path.join(val_dir, 'nv')
    os.mkdir(nv)
    mel = os.path.join(val_dir, 'mel')
    os.mkdir(mel)
    bkl = os.path.join(val_dir, 'bkl')
    os.mkdir(bkl)
    bcc = os.path.join(val_dir, 'bcc')
    os.mkdir(bcc)
    akiec = os.path.join(val_dir, 'akiec')
    os.mkdir(akiec)
    vasc = os.path.join(val_dir, 'vasc')
    os.mkdir(vasc)
    df = os.path.join(val_dir, 'df')
    os.mkdir(df)

    # UNCOMMENT IF HAM10000/ISIC images are in .jpg form. This will take a hot minute
    
    # image_part1_path = '../Skin_Lesions/archive/HAM10000_images_part_1/'
    # for img in os.listdir(image_part1_path):
    #     print(image_part1_path+img[:-4] + ".png")
    #     input_image = Image.open(image_part1_path+img).convert('RGB')
    #     input_image = input_image.resize((224, 224))
    #     input_image.save(image_part1_path+img[:-4] + ".png", "PNG")
    #     os.remove(image_part1_path+img)
    #         #print(f2[key])
    # image_part2_path = '../Skin_Lesions/archive/HAM10000_images_part_2/'
    # for img in os.listdir(image_part2_path):
    #     input_image = Image.open(image_part2_path+img).convert('RGB')
    #     input_image = input_image.resize((224, 224))
    #     input_image.save(image_part2_path+img[:-4] + ".png", "PNG")
    #     os.remove(image_part2_path+img)
    #         #print(f2[key])

    return val_dir, training_dir