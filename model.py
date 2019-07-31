from generator import generate_batches
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Input,Embedding,LSTM,Dropout
from keras import metrics
from keras.layers.merge import add
from keras.optimizers import Adam
from load_caption_encodings import load_caption_encodings
from load_word_index import load_word_index
import numpy as np
import os

train_image_captions,val_image_captions,train_im_encs,val_im_encs=load_caption_encodings()
word_to_index,index_to_word,new_vocab_length,max_length=load_word_index()

glove_enc_length=200
glove_dir="./dataset/glove_data/glove.6B.200d.txt"
f=open(glove_dir,encoding="utf-8")
glove_encodings={}
for line in f:
  values=line.split()
  word=values[0]
  enc=np.array(values[1:],dtype="float32")
  glove_encodings[word]=enc

index_to_glove=np.zeros((new_vocab_length,glove_enc_length))
for word,index in word_to_index.items():
  if index in glove_encodings:
    index_to_glove[index]=glove_encodings[word]


input1=Input((2048,))
l11=Dense(512,activation='relu')(input1)
l12=Dropout(0.5)(l11)
l13=Dense(256,activation='relu')(l12)

input2=Input((max_length,))
l21=Embedding(new_vocab_length,glove_enc_length,mask_zero=True)(input2)
l22=Dropout(0.5)(l21)
l23=LSTM(256)(l22)

dec1=add([l13,l23])
dec2=Dense(2048,activation='relu')(dec1)
dec3=Dense(new_vocab_length,activation='softmax')(dec2)

model_final=Model([input1,input2],[dec3])
model_final.layers[3].set_weights([index_to_glove])
model_final.layers[3].trainable=False

print("Model Summary:")
print(model_final.summary())

model_final.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=[metrics.categorical_accuracy])

epochs = 100
number_pics_per_batch = 4
steps_train = len(train_image_captions)//number_pics_per_batch
steps_val = len(val_image_captions)//number_pics_per_batch

if not os.path.isdir('./checkpoints/'):
	os.mkdir('./checkpoints/')

checkpoint = ModelCheckpoint('./checkpoints/model_{val_loss:.4f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

generator_train = generate_batches(train_image_captions,word_to_index,train_im_encs,max_length,number_pics_per_batch,new_vocab_length)
generator_val = generate_batches(val_image_captions,word_to_index,val_im_encs,max_length,number_pics_per_batch,new_vocab_length)
model_final.fit_generator(generator_train, 
							epochs=epochs, 
							steps_per_epoch=steps_train,
							validation_data=generator_val,
							validation_steps=steps_val,
							callbacks=[checkpoint],
							shuffle=True)

model_json_conv = model_final.to_json()
with open("./checkpoints/model.json", "w") as json_file:
  json_file.write(model_json_conv)