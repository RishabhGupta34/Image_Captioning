from keras.models import  model_from_json
import os
import sys

def load_model():
	files=os.listdir('./checkpoints/')
	json_file_name=None
	for file in files:
		if('.json' in file):
			json_file_name=file
			break

	files=[file for file in files if '.h5' in file]

	if(not os.path.exists('./checkpoints/'+json_file_name)):
		sys.exit("Error! Model Not Trained. Please train the model and then continue")
	json_file = open('./checkpoints/'+json_file_name, 'r')
	loaded_model_json = json_file.read()
	json_file.close()


	if(len(files)==0):
		sys.exit("Error! Model Not Trained. Please train the model and then continue")

	files.sort()

	model_final = model_from_json(loaded_model_json)
	model_final.load_weights('./checkpoints/'+files[0])
	return model_final