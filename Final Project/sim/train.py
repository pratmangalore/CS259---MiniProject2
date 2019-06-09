import sys
import time
import keras

OPTYPE_ERROR = 13
LHR_MAX_ENTRY = 64
GHR_MAX_LEN = 100
LHR_MAX_LEN = 16

def read_traces(path):
	with open(path,"r") as raw_data:
		line_number = 0
		x_data = []
		y_data = []
		ghr = [i%2 for i in range(GHR_MAX_LEN)]
		lhr = {}
		start_time = time.time()
		for raw_line in raw_data:
			line = raw_line.split(", ")
			line[3] = line[3][0]
	
			if(int(line[0]) == OPTYPE_ERROR):
				continue

			ghr.append(line[3])

			if(len(ghr) > GHR_MAX_LEN):
				del ghr[0]

			feature_set = line[0] + line[1] + line[2]

			if(feature_set not in lhr):
				lhr[feature_set] = [i%2 for i in range(LHR_MAX_LEN)]
			if(len(lhr[feature_set]) > LHR_MAX_LEN):
				del lhr[feature_set][0]

			lhr[feature_set].append(line[3])

			x_data.append([feature_set,''.join([str(elem) for elem in ghr]) ,''.join([str(elem) for elem in lhr[feature_set]])])
			y_data.append(line[3])

			print(x_data[line_number],y_data[line_number])
			print()
			line_number = line_number + 1

			if(line_number > 5):
				break
			if(line_number % 100000 == 0):
				end_time = time.time()
				print("Time taken := " + str(end_time - start_time))
				start_time = end_time
			
	return x_data,y_data

path = "../traces/LONG_MOBILE-1.bt9.trace.gz.txt"
x_data,y_data = read_traces(path)



