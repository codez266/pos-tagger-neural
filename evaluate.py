from sklearn.metrics import classification_report
import numpy as np
import pdb

def report(gold, pred, labels):
	y_pred = np.loadtxt(pred)
	y_pred = list(map(lambda x:int(x), y_pred))
	y_gold = np.loadtxt(gold, dtype=np.string_)
	#y_gold = list(map(lambda x:int(x), y_gold))
	#pdb.set_trace()
	tags = np.loadtxt(labels, dtype=np.string_)
	taglabels = {}
	ord_labels = []
	idx = 0
	for w in tags:
		if w not in taglabels:
			taglabels[w] = idx
			ord_labels.append(w)
			idx = idx + 1
	y_gold = list(map(lambda x: taglabels[x], y_gold))
	#pdb.set_trace()
	print(classification_report(y_gold, y_pred, target_names=ord_labels))
	#print(classification_report(y_gold, y_pred))
report('brown_Y_test', 'class.txt', 'brown_Y')
