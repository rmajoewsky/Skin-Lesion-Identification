import pandas as pd
import numpy as np
#import keras
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from pathlib import Path

from Load_Data import load_data


def create_train_and_val():
    df_data = pd.read_csv('../Skin_Lesions/archive/HAM10000_metadata.csv')
    #print(df_data.head())
    df_data2  = pd.read_csv('../Skin_Lesions/zr7vgbcyr2-1/metadata.csv')
    #print(df_data2.head())
    df_data2.drop_duplicates('lesion_id')
    #df_data2 = df_data2['lesion_id', 'img_id', 'diagnostic', 'biopsied', 'age', 'gender']
    df_data2 = df_data2.rename(columns={"img_id": "image_id", "diagnostic": "dx", 'gender': "sex"})
    cols = ['lesion_id','image_id','dx','age','sex']
    df_data2 = df_data2[cols]  
    df_data = df_data[cols]
    df_data2 = df_data2.replace(['BCC', 'ACK', 'MEL', 'NEV', 'SEK', 'SCC'],['bcc', 'akiec', 'mel', 'nv', 'bkl', 'akiec'])
    df_data2 = df_data2.fillna('N/A')
    print(df_data2.head())

    #print(df_data2.groupby('patient_id').count())

    df = df_data.groupby('lesion_id').count()
    
    # filter out lesion_id's that have only one image associated with it
    df = df[df['image_id'] == 1]
    df.reset_index(inplace=True)

    # identify lesion IDs with duplicate images
    df[df.lesion_id.duplicated(keep=False)]
    df = df_data[df_data.lesion_id.isin(df.lesion_id)]
    
    y = df['dx']
    _, df_val = train_test_split(df, test_size=0.20, random_state=101, stratify=y)
    m = df_data.image_id.isin(df_val.image_id)
    df_train = df_data[~m]
    df_data.set_index('image_id', inplace=True)
    
    y2 = df_data2['dx']
    _, df_val2 = train_test_split(df_data2, test_size=0.20, random_state=101, stratify=y2)
    m = df_data2.image_id.isin(df_val2.image_id)
    df_train2 = df_data2[~m]
    df_data2.set_index('image_id', inplace=True)
    

    return df_train, df_val, df_data, df_train2, df_val2, df_data2 
    
def transfer_images_2_folders(val_dir, training_dir):
    df_train, df_val, df_data, df_train2, df_val2, df_data2 = create_train_and_val()
    #read in images
    f1 = '../Skin_Lesions/archive/HAM10000_images_part_1'
    f2 = '../Skin_Lesions/archive/HAM10000_images_part_2'
    part1 = '../Skin_Lesions/zr7vgbcyr2-1/images/imgs_part_1'

    part2 = '../Skin_Lesions/zr7vgbcyr2-1/images/imgs_part_2'
    part3 = '../Skin_Lesions/zr7vgbcyr2-1/images/imgs_part_3'
    
    #combine folders into dictionaries
    def Merge(dict1, dict2):
        return(dict2.update(dict1))
    
    f1_df = {file[:-4]: f1 + "/" + file for file in os.listdir(f1)}
    f2_df = {file[:-4]: f2 + "/" + file for file in os.listdir(f2)}
    part1_df = {file[:-4]: part1 + "/" + file for file in os.listdir(part1)}
    part2_df = {file[:-4]: part2 + "/" + file for file in os.listdir(part2)}
    part3_df = {file[:-4]: part3 + "/" + file for file in os.listdir(part3)}

    Merge(f1_df, f2_df)
    Merge(part1_df, part2_df)
    Merge(part2_df, part3_df)
    #print(part3_df)
    
    train_list = list(df_train['image_id'])
    print(train_list)
   
    val_list = list(df_val['image_id'])
    print(val_list)
    
    for image in train_list:
        shutil.copyfile(f2_df[image], os.path.join(training_dir, df_data.loc[image,'dx'], image + '.png'))

    for image in val_list:
       
        shutil.copyfile(f2_df[image], os.path.join(val_dir, df_data.loc[image,'dx'], image + '.png'))

    train_list2 = list(df_train2['image_id'])
    val_list2 = list(df_val2['image_id'])

    for image in train_list2:
        shutil.copyfile(part3_df[image[:-4]], os.path.join(training_dir, df_data2.loc[image,'dx'], image))

    for image in val_list2:  
        shutil.copyfile(part3_df[image[:-4]], os.path.join(val_dir, df_data2.loc[image,'dx'], image))

# Handles a dataset that was added later, TODO streamline this
def dummy_set():
    df_data3 = pd.read_csv('../Skin_Lesions/ISIC-images/metadata.csv')
    print(df_data3.head())
    df_data3 = df_data3.dropna()
    val_dir = '../Skin_Lesions/base_dir/val_dir'
    training_dir = '../Skin_Lesions/base_dir/train_dir'

    # image_part1_path = '../Skin_Lesions/ISIC-images/Images/'
    # for img in os.listdir(image_part1_path):
    #     #print(image_part1_path+img[:-4] + ".png")
    #     print(img[-3:])
    #     if img[-3:] == 'jpg':
            
    #         input_image = Image.open(image_part1_path+img).convert('RGB')
    #         input_image = input_image.resize((224, 224))
    #         input_image.save(image_part1_path+img[:-4] + ".png", "PNG")
    #         os.remove(image_part1_path+img)

    # df = df_data3.groupby('lesion_id').count()
    
    # 
    # df = df[df['image_id'] == 1]
    # df.reset_index(inplace=True)
    # df[df.lesion_id.duplicated(keep=False)]
    # df = df_data3[df_data3.lesion_id.isin(df.lesion_id)]
    # # #print(df.head())

    df_data3 = df_data3.drop_duplicates()

    y3 = df_data3['dx']
    _, df_val3 = train_test_split(df_data3, test_size=0.20, random_state=101, stratify=y3)
    m = df_data3.image_id.isin(df_val3.image_id)
    df_train3 = df_data3[~m]
    df_data3.set_index('image_id', inplace=True)
    part4 = '../Skin_Lesions/ISIC-images/Images/'

    f3_df = {file[:-4]: part4 + "/" + file for file in os.listdir(part4)}
    print(df_train3.head())
    train_list3 = list(df_train3['image_id'])
    val_list3 = list(df_val3['image_id'])
   
    for image in train_list3:
        shutil.copyfile(f3_df[image], os.path.join(training_dir, df_data3.loc[image,'dx'], image + '.png'))

    for image in val_list3:
        shutil.copyfile(f3_df[image], os.path.join(val_dir, df_data3.loc[image,'dx'], image + '.png'))
    
    
def test_prints():
   
    print(len(os.listdir('base_dir/val_dir/nv')))
    print(len(os.listdir('base_dir/val_dir/mel')))
    print(len(os.listdir('base_dir/val_dir/bkl')))
    print(len(os.listdir('base_dir/val_dir/bcc')))
    print(len(os.listdir('base_dir/val_dir/akiec')))
    print(len(os.listdir('base_dir/val_dir/vasc')))
    print(len(os.listdir('base_dir/val_dir/df')))

    

    


def main():
    val_dir, training_dir = load_data()
    #val_dir = '../Skin_Lesions/base_dir/val_dir'
    #training_dir = '../Skin_Lesions/base_dir/train_dir'
    
    transfer_images_2_folders(val_dir, training_dir)
    #test_prints()
    
    

    

if __name__ == "__main__":
    main()