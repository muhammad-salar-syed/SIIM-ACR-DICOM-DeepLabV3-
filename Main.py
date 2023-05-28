from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
from skimage import io
from PIL import Image
import glob

train_img_path=glob.glob('./train/dicom_files/*')[:8560]
train_mask_path=glob.glob('./train/images/1024/mask/*')[:8560]
val_img_path=glob.glob('./train/dicom_files/*')[8560:10700]
val_mask_path=glob.glob('./train/images/1024/mask/*')[8560:10700]

seed=24
batch_size=8

def ImageGenerator(img_list, mask_list, batch_size):

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < len(img_list):
            limit = min(batch_end, len(img_list))
            
            X=[]
            Y=[]
            for i in img_list[batch_start:limit]:
                ds = dicom.dcmread(i)
                img=ds.pixel_array
                r_img = Image.fromarray(img)
                r_img = np.array(r_img.resize((272,192)))/255.
                S_img = np.stack((r_img,)*3, axis=-1)
                X.append(S_img)
             
            for j in mask_list[batch_start:limit]:
                mask=io.imread(j)
                r_mask = Image.fromarray(mask)
                r_mask = np.array(r_mask.resize((272,192)))/255.
                Y.append(r_mask)
                
            images = np.array(X)
            masks = np.expand_dims(np.array(Y), axis=3)

            yield (images,masks)     

            batch_start += batch_size   
            batch_end += batch_size


train_img_gen = ImageGenerator(train_img_path,train_mask_path,batch_size)
val_img_gen = ImageGenerator(val_img_path,val_mask_path,batch_size)

steps_per_epoch = len(train_img_path)//batch_size
val_steps_per_epoch = len(val_img_path)//batch_size

img, mask = train_img_gen.__next__()


def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x

def ASPP(inputs):
    """ Image Pooling """
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    """ 1x1 conv """
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    """ 3x3 conv rate=6 """
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    """ 3x3 conv rate=12 """
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    """ 3x3 conv rate=18 """
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def deeplabv3_plus(shape):

    inputs = Input(shape)

    """ Encoder """
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    image_features = encoder.get_layer("conv4_block6_out").output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    x_b = encoder.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])
    x = SqueezeAndExcite(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model

input_shape=(192,272,3)
model=deeplabv3_plus(input_shape)
model.summary()

import segmentation_models as sm
metrics = [sm.metrics.IOUScore(threshold=0.5)]
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
callbacks_list = [early_stop]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=60,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch,
          callbacks=callbacks_list)

model.save('./dicom_deeplabv3_plus.hdf5')



