from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical

def generate_batches(image_captions,word_to_ix,image_encs,max_length,num_photo_per_batch,new_vocab_length):
  X1,X2,y=list(),list(),list()
  n=0
  while 1:
    for key,desc_list in image_captions.items():
      n+=1
      ph_enc=image_encs[key+".jpg"]
      for desc in desc_list:
        sequences=[word_to_ix[word] for word in desc.split() if word in word_to_ix]
        for i in range(1,len(sequences)):
          in_seq,out_seq=sequences[:i],sequences[i]
          in_seq=pad_sequences([in_seq],max_length)[0]
          out_seq=to_categorical([out_seq],num_classes=new_vocab_length)[0]
          X1.append(ph_enc)
          X2.append(in_seq)
          y.append(out_seq)
      if(n==num_photo_per_batch):
        yield(([np.array(X1),np.array(X2)],np.array(y)))
        X1,X2,y=list(),list(),list()
        n=0