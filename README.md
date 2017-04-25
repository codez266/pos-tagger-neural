# POS Tagger using Neural Net

A simple pos tagger using neural nets. The corpora used was brown corpus.
The library for neural nets is keras.

## Architecture
I used a simple feed forward neural network with a single hidden layer of 100 neurons,
and a multiclass output for 472 tags.

## Stats of the first few tags from report.txt on brown corpus
				 precision    recall  f1-score   support

		  b'AT'       0.92      0.93      0.93     17192
	   b'NP-TL'       0.66      0.45      0.53       334
	   b'NN-TL'       0.77      0.76      0.76      1056
	   b'JJ-TL'       0.81      0.62      0.70       276
		 b'VBD'       0.89      0.97      0.93     10607
		  b'NR'       0.95      0.78      0.86       329
		  b'NN'       0.92      0.93      0.92     25750
		  b'IN'       0.80      0.89      0.85     19273
		 b'NP$'       0.43      0.24      0.31       597
		  b'JJ'       0.82      0.88      0.85      9751
		  b'``'       0.78      0.86      0.82      3353
		  b"''"       0.91      0.63      0.74      3344
		  b'CS'       0.92      0.84      0.88      4201
		 b'DTI'       1.00      0.83      0.91       482
		 b'NNS'       0.97      0.93      0.95      7076
