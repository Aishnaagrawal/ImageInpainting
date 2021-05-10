import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from flask import Flask, render_template, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

app = Flask(__name__,template_folder='templates')

def createmask(img,mask):
  img = img.astype(np.uint8)
  masked_image = np.copy(img)
  for l in range(len(img)):
    masked_image[l] = cv2.bitwise_and(img[l], mask)
  return masked_image/255

class Generator():
  # UNET model
  def train_model(self, input_size=(256,256,3)):
    inputs = tf.keras.layers.Input(input_size)
    acti_function = "relu"
    padding = "same"
    filters = 32
    kernel_size = (3,3)
    pool_size = (2,2)
    up_kernel = (2,2)
    up_stride = (2,2)
    # encoder 
    conv1, pooling1 = self.Convulation_layer(filters,kernel_size, pool_size, acti_function, padding, inputs)
    conv2, pooling2 = self.Convulation_layer(filters*2, kernel_size, pool_size, acti_function, padding, pooling1)
    conv3, pooling3 = self.Convulation_layer(filters*4, kernel_size, pool_size, acti_function,padding, pooling2) 
    conv4, pooling4 = self.Convulation_layer(filters*8, kernel_size, pool_size, acti_function,padding, pooling3) 
    # decoder 
    conv5, up6 = self.Up_Convulation_layer(filters*16, filters*8, kernel_size, up_kernel, up_stride, acti_function, padding, pooling4, conv4)
    conv6, up7 = self.Up_Convulation_layer(filters*8, filters*4, kernel_size, up_kernel, up_stride, acti_function,padding, up6, conv3)
    conv7, up8 = self.Up_Convulation_layer(filters*4, filters*2, kernel_size, up_kernel, up_stride, acti_function, padding, up7, conv2)
    conv8, up9 = self.Up_Convulation_layer(filters*2,filters, kernel_size, up_kernel, up_stride, acti_function,padding, up8, conv1)
    conv9 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv9)
    outputs = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
    return tf.keras.models.Model(inputs=[inputs], outputs=[outputs]) 
    
  def Convulation_layer(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    pooling = tf.keras.layers.MaxPooling2D(pool_size)(conv)
    return conv, pooling

  def Up_Convulation_layer(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    up = tf.keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
    up = tf.keras.layers.concatenate([up, shared_layer], axis=3)
    return conv, up


@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def result():
    uploaded_file = request.files['input-img']
    result = request.form
    print(result)
    option = result["masks"]
    print(type(option))
    if uploaded_file.filename != '':
        uploaded_file.save("./static/Output/input-image.png")

    tf.keras.backend.clear_session()
    model = Generator().train_model()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.load_weights("d1.h5")
    mask = np.full((256,256,3), 255, np.uint8)

    # Creating a rectangular 64*64 mask
    for i in range(3):
        for j in range(256):
            for k in range(256):
              if option == "1":
                if j>50 and 114>j and k>50 and 114>k:
                    mask[j][k][i] = 1
              elif option == "2":
                if j>96 and 160>j and k>96 and 160>k:
                    mask[j][k][i] = 1
              elif option == "3":
                if j>96 and 160>j and k>114 and 178>k:
                    mask[j][k][i] = 1
              elif option == "4":
                if j>114 and 178>j and k>114 and 178>k:
                    mask[j][k][i] = 1


    pic1 = np.asarray(Image.open("./static/Output/input-image.png")).reshape(1,256,256,3)
    x_mask1 = createmask(pic1,mask)
    yhat1 = model.predict(x_mask1).reshape(256,256,3)
    image = Image.fromarray((yhat1*255).astype(np.uint8))
    image.save('./static/Output/output-image.png')
    image = Image.fromarray(((x_mask1.reshape(256,256,3))*255).astype(np.uint8))
    image.save('./static/Output/masked-image.png')
    return render_template("result.html")

if __name__ == '__main__':
   app.run(debug = True)

