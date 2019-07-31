from nltk.probability import FreqDist
from read_captions import read_captions
from clean_captions import clean_captions
from image_to_encoding import images_to_enc
import os
import pickle

train_image_captions=read_captions("./dataset/annotations/captions_train2014.json",1) 
val_image_captions=read_captions("./dataset/annotations/captions_val2014.json",2) 

train_image_captions=clean_captions(train_image_captions)
val_image_captions=clean_captions(val_image_captions)
if not os.path.isdir('./processed_data/'):
	os.mkdir('./processed_data/')

print("No. of training set images: ",len(train_image_captions.keys()))
print("No. of validation set images: ",len(val_image_captions.keys()))

def make_vocab(image_captions,threshold):
  fq=FreqDist([i for key in list(image_captions.keys()) for w in image_captions[key] for i in w.split()])
  vocab=set([w for w,cnt in list(fq.items()) if cnt>=threshold])
  return len(list(fq.items())),vocab,len(vocab)


old_vocab_length,vocab,new_vocab_length=make_vocab(train_image_captions,5)
print("Old Vocabulary Length: ",old_vocab_length)
new_vocab_length=new_vocab_length+1
print("New Vocabulary Length: ",new_vocab_length)


def word_index(vocab):
  word_to_index={}
  index_to_word={}
  i=1
  for w in vocab:
    word_to_index[w]=i
    index_to_word[i]=w
    i+=1
  return word_to_index,index_to_word

word_to_index,index_to_word=word_index(vocab)
# print(len(word_to_ix))

def calc_max_length(image_captions):
  return max([max([len(desc.split()) for desc in image_captions[key]]) for key in image_captions.keys()])

max_length=calc_max_length(train_image_captions)

train_im_path="./dataset/train2014/"
val_im_path="./dataset/val2014/"

print("Encoding Train Images")
train_im_encs=images_to_enc(list(train_image_captions.keys()),train_im_path)
print("Encoded Train Images")

print("Encoding Validation Images")
val_im_encs=images_to_enc(list(val_image_captions.keys()),val_im_path)
print("Encoded Validation Images")

# print(len(train_im_encs),len(val_im_encs))


print("Saving Processed Dataset!")
if not os.path.isdir('./processed_data/'):
	os.mkdir('./processed_data/')
with open('./processed_data/new_vocab_length.pickle', 'wb') as handle:
    pickle.dump(new_vocab_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./processed_data/max_length.pickle', 'wb') as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./processed_data/train_image_captions.pickle', 'wb') as handle:
    pickle.dump(train_image_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./processed_data/val_image_captions.pickle', 'wb') as handle:
    pickle.dump(val_image_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./processed_data/word_to_index.pickle', 'wb') as handle:
    pickle.dump(word_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./processed_data/index_to_word.pickle', 'wb') as handle:
    pickle.dump(index_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./processed_data/train_im_encs.pickle', 'wb') as handle:
    pickle.dump(train_im_encs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./processed_data/val_im_encs.pickle', 'wb') as handle:
    pickle.dump(val_im_encs, handle, protocol=pickle.HIGHEST_PROTOCOL)
