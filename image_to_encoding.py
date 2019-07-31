from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import numpy as np
import cv2

def load_image(image_path):
  img=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
  img=cv2.resize(img,(299,299))
  img=np.reshape(img,(1,299,299,3))
  img=preprocess_input(img)
  return img

def encode_images(image_encoder_model,image_path):
  img=load_image(image_path)
  enc=image_encoder_model.predict(img)
  enc=np.reshape(enc,enc.shape[1])
  return enc

def encode_image(image_encoder_model,image_path):
  img=load_image(image_path)
  enc=image_encoder_model.predict(img)
  return enc

def images_to_enc(images,images_path):
  image_encoder_model = InceptionV3(weights='imagenet')
  image_encoder_model=Model(image_encoder_model.input,image_encoder_model.layers[-2].output)
  image_encs={}
  for i in range(len(images)):
    if(i%2000==0):
      print("{} images out of {} encoded successfully".format(i,len(images)))
    image_encs[images[i]]=encode_images(image_encoder_model,images_path+images[i]+".jpg")
  return image_encs


def image_to_enc(image):
  image_encoder_model = InceptionV3(weights='imagenet')
  image_encoder_model=Model(image_encoder_model.input,image_encoder_model.layers[-2].output)
  enc=encode_image(image_encoder_model,image)
  return enc