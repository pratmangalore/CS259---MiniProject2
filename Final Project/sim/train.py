import time
import keras
import numpy as np
import pandas as pd

chunk_size = 50

def run(path):
	reader = pd.read_csv(path,chunksize = chunk_size,header=0)

	"""
	model = Sequential()
	model.add(Conv1D(20,kernel_size = (8),activation = 'relu',input_shape = (chunk_size,)))
	model.add(MaxPooling1D(pool_size = 2))
	model.add(Conv1D(50,kernel_size = (8),activation = 'relu'))
	model.add(MaxPooling1D(pool_size = 2))
	model.add(Dense(500,activation = 'relu'))
	model.add(Dense(1,activation = 'softmax'))
	"""

	for chunk in reader:
		start_time = time.time()
		X_batch = []
		y_batch = []
		for row in chunk.itertuples():
			X_elem = []
			X_elem_str = list(row.X)
			for elem in X_elem_str:
				X_elem.append(int(elem))
			X_batch.append(X_elem)
			y_batch.append(int(row.y))
		X_batch = np.asarray(X_batch)
		y_batch = np.asarray(y_batch)
		end_time = time.time()
		print(end_time - start_time)
		break
		
	
path = "../traces/data.csv"
model = train(path)