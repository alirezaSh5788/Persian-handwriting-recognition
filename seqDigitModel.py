import keras
import glob
from tqdm import tqdm
import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Dropout,BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt


def build_model(inputs):
    x = inputs

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                     input_shape=input))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    return model


num_classes = 10
EPOCHS = 20
BS = 32

train_dirs = glob.glob("./dataset/digit_folders/*")
train_dirs.sort()

data = []
labels = []
for train_dir in tqdm(train_dirs):
    imgPaths = glob.glob(train_dir + "/*.jpg")
    imgPaths.sort()
    for imgPath in tqdm(imgPaths):
        image = load_img(imgPath, target_size=(28, 28), grayscale=True)
        image = img_to_array(image) 
        data.append(image)

        label = imgPath.split(os.path.sep)[-2]
        label = int(label)
        labels.append(label)

data = np.array(data, dtype=np.float) / 255.
labels = np.array(labels)

train_input, valid_input, train_target, valid_target = train_test_split(data,
                                                                        labels,
                                                                        test_size=0.25,
                                                                        random_state=123)

train_target = to_categorical(train_target, num_classes)
valid_target = to_categorical(valid_target, num_classes)

aug = ImageDataGenerator( rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)

input = (28, 28, 1)
model = build_model(input)

opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])

checkpoint = ModelCheckpoint(filepath="seq_digit_model.h5",
                             monitor="val_acc",
                             verbose=1,
                             save_best_only=True)

training_log = model.fit_generator(aug.flow(train_input, train_target, batch_size=BS),
                                   validation_data=(valid_input, valid_target),
                                   epochs=EPOCHS,
                                   callbacks=[checkpoint],
                                   verbose=2)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(EPOCHS), training_log.history["loss"], label="train_loss")
plt.plot(np.arange(EPOCHS), training_log.history["acc"], label="train_acc")
plt.plot(np.arange(EPOCHS), training_log.history["val_loss"], label="val_loss")
plt.plot(np.arange(EPOCHS), training_log.history["val_acc"], label="val_acc")
plt.xlabel("Epochs")
plt.ylabel("loss/accuracy")
plt.title("training plot")
plt.legend(loc="center right")
plt.savefig("digit_training_plot.png")

model.load_weights("seq_digit_model.h5")
model.save('seq_digit_model')
