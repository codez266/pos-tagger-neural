import numpy as np
import logging
import pickle
import pdb
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'

def vectorize_y(filein, fileout):
	idx = 0
	tags = {}
	with open(filein) as fin:
		for line in fin:
			tag = line.strip()
			if tag not in tags:
				tags[tag] = idx
				idx = idx + 1
	l = idx
	fout = open(fileout, 'w')
	#vect = np.zeros((1, l), dtype=int)
	with open(filein) as fin:
		for line in fin:
			v = np.zeros(l, dtype=int)
			index = tags[line.strip()]
			v[index] = 1
			st = ''
			for el in v:
				st = st + ' ' + str(el)
			fout.write(st + '\n')
	fout.close()
	#pdb.set_trace()
	#pickle.dump(vect[1:,:], open('tags.model', 'wb'))

def vectorize_X(filein, fileout, w2v):
	logger = logging.getLogger()
	print('Loading word vectors')
	model = Word2Vec.load_word2vec_format(w2v, binary=True)  # C binary format
	print('Loaded word vectors')
	fout = open(fileout, 'w')
	with open(filein) as fin:
		for l in fin:
			line = l.strip().split()
			vect = np.array([], dtype=np.float64)
			for w in line:
				if w in model:
					vect = np.append(vect, model[w])
				else:
					vect = np.append(vect, np.zeros(300))
			outp = ''
			outp = ' '.join(map(lambda x:str(x), vect))
			fout.write(outp.strip()+' '+'\n')
	fout.close()

def gen_examples(filein, fileout):
	with open(filein) as fin:
		foutx = open(fileout+'_X', 'w')
		fouty = open(fileout+'_Y', 'w')
		for l in fin:
			line = l.strip().split()
			for i, w in enumerate(line):
				try:
					word, tag = w.split("_")
					tok = ['_' for i in range(0,5)]
					for j in range(0,5):
						if i + j - 2 >= 0 and i + j - 2 < len(line):
							if j == 2:
								tok[2] = word
							else:
								tok[j] = line[i+j-2].split('_')[0]
					lineout = ' '.join(tok)
					foutx.write(lineout+'\n')
					fouty.write(tag+'\n')
				except Exception:
					pass
		foutx.close()
		fouty.close()

def tagger(filex, filey):
	X_train = np.loadtxt(filex+'_train')
	Y_train = np.loadtxt(filey+'_train')
	l = len(X_train)
	test = int(0.1*l)
	train = int(0.9*l)
	X_test = X_train[train:]
	Y_test = Y_train[train:]
	X_train = X_train[0:train]
	Y_train = Y_train[0:train]
	model = Sequential()
	model.add(Dense(100, input_dim=1500))
	model.add(Activation('tanh'))
	model.add(Dense(472, init='uniform'))
	model.add(Activation('softmax'))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit(X_train, Y_train, nb_epoch=5, batch_size=16)
	score = model.evaluate(X_test, Y_test, batch_size=16)
	print(score)

#gen_examples('Brown_train.txt', 'brown')
#vectorize_y('train_Y', 'tags-vector')
#vectorize_X('brown_X', 'brownV_X', '../../btp/GoogleNews-vectors-negative300.bin')
tagger('brown', 'tags')
