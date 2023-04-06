from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def train_and_predict(model):
    train_path = '../Skin_Lesions/base_dir/train_dir'
    val_path = '../Skin_Lesions/base_dir/val_dir'
    train_file_count = sum(len(files) for _, _, files in os.walk(train_path))
    #print(train_file_count)
    val_file_count = sum(len(files) for _, _, files in os.walk(val_path))
    #print(val_file_count)
    
    initial_lr = 1e-4
    epochs = 30
    batch_size = 32
    image_size = 224
    decay = initial_lr / epochs
    
    file_path = "model.h5"

    # Weight so model is more sensitive to melanoma
    class_weights = {
        0: 1.0, # akiec
        1: 1.0, # bcc
        2: 1.0, # bkl
        3: 1.0, # df
        4: 3.0, # mel 
        5: 1.0, # nv
        6: 1.0, # vasc
    }

    datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=batch_size)


    valid_batches = datagen.flow_from_directory(val_path,
                                            target_size=(image_size,image_size),
                                            batch_size=batch_size)

    
    test_batches = datagen.flow_from_directory(val_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)
    
   #print(valid_batches.class_indices)

    opt = Adam(learning_rate=initial_lr, decay=initial_lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    
    fit = model.fit(train_batches,
        steps_per_epoch=train_file_count // batch_size,
        class_weight=class_weights,
        validation_data=valid_batches,
        validation_steps=val_file_count // batch_size,
        epochs=epochs)
    preds = model.predict(test_batches)
    # for each image in the testing set we need to find the index of the
    # label with the corresponding largest predicted probability
    preds = np.argmax(preds, axis=1)
    test_labels = test_batches.classes
    print(classification_report(test_labels, preds))
    # show a nicely formatted classification report
    print(confusion_matrix(test_labels, preds))
    # show a nicely formatted classification report
    print(accuracy_score(test_labels, preds))

    model.save('skin_lesion_classifier.model', save_format="h5")

    



def create_model():
    
    # create base model, freeze layers in base model so they don't get updated in first training
    # practice
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    for layer in base_model.layers: 
        layer.trainable = False

    # create head model, citation for (initial) model structure: 
    # https://ai.plainenglish.io/blood-face-detector-in-python-part-1-machine-learning-and-deep-learning-classification-project-74aba7067e50
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(7, activation="softmax")(head_model)

    # head model on top of base model
    model = Model(inputs=base_model.input, outputs=head_model)

    # train the model, make predictions on testing set 
    train_and_predict(model)


def main():
    create_model()
    
if __name__ == "__main__":
    main()