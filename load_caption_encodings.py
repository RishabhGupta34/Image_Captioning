import pickle

def load_caption_encodings():
	with open('./processed_data/train_image_captions.pickle', 'rb') as handle:
	    train_image_captions=pickle.load(handle)
	with open('./processed_data/val_image_captions.pickle', 'rb') as handle:
	    val_image_captions=pickle.load(handle)
	with open('./processed_data/train_im_encs.pickle', 'rb') as handle:
	    train_im_encs=pickle.load(handle)
	with open('./processed_data/val_im_encs.pickle', 'rb') as handle:
	    val_im_encs=pickle.load(handle)
	return train_image_captions,val_image_captions,train_im_encs,val_im_encs