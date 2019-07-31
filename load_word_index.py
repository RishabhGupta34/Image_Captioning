import pickle

def load_word_index():
	with open('./processed_data/word_to_index.pickle', 'rb') as handle:
	    word_to_index=pickle.load(handle)
	with open('./processed_data/index_to_word.pickle', 'rb') as handle:
	    index_to_word=pickle.load(handle)
	with open('./processed_data/new_vocab_length.pickle', 'rb') as handle:
	    new_vocab_length=pickle.load(handle)
	with open('./processed_data/max_length.pickle', 'rb') as handle:
	    max_length=pickle.load(handle)
	return word_to_index,index_to_word,new_vocab_length,max_length