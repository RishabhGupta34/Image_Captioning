import sys
import os
from image_to_encoding import image_to_enc
from load_saved_model import load_model
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from load_word_index import load_word_index
import matplotlib.pyplot as plt


word_to_index,index_to_word,new_vocab_length,max_length=load_word_index()

if(len(sys.argv)!=2):
	sys.exit("No folder or image detected in the system arguments. Try Again!")


folder = sys.argv[1]


model_final=load_model()
def beam_search_predictions(model_final,image, beam_index = 3):
    start = [word_to_index["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = model_final.predict([image,par_caps])
            word_preds = np.argsort(preds[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    final_captions = []
    for i in start_word:
      caption=[index_to_word[j] for j in i[0]]
      final=[]
      for word in caption:
        if word=='startseq':
          continue
        elif word=='endseq':
          break
        else:
          final.append(word)
      final_captions.append(" ".join(final))
    return final_captions

def greedySearch(model_final,photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_to_index[w] for w in in_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model_final.predict([photo,sequence])
        yhat = np.argmax(yhat[0])
        word = index_to_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

if(not os.path.isdir(folder)):
	image = image_to_enc(folder)
	x=plt.imread(folder)
	plt.imshow(x)
	plt.show()
	print("Predicted Caption: ",beam_search_predictions(model_final,image))
else:
	if(not os.path.exists(folder)):
		sys.exit("Folder path incorrect. Try Again!")
	files=os.listdir(folder)
	if(folder[-1]!='/'):
		folder+='/'
	predictions=[]
	for img in range(len(files)):
		pic = files[img]
		image = image_to_enc(folder+pic)
		predictions.append(beam_search_predictions(model_final,image))
		print("{} of {} completed".format(img,len(files)))

	pd.DataFrame({"Image_Name":files,"Predicted Captions":predictions}).to_csv("Predicitons.csv")
	print("CSV File as 'Predictions.csv' created with all predictions")


