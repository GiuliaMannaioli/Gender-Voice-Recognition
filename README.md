# Gender-Voice-Recognition
This project consists in gender identification from high-level features. 
The dataset consists of synthetic speaker embeddings that represent the acoustic 
characteristics of a spoken utterance (the dataset consists of synthetic samples that behave 
similarly to real speaker embeddings).
A speaker embedding is a small-dimensional, fixed sized representation of an utterance.
Speakers belong to four different age groups. The age information, however, is not available.
The training set consists of 3000 samples per class, whereas the test set contains 2000 
samples per class. 
Classes are balanced.
Each sample (each row) corresponds to a different speaker, and contains 12 features followed by the gender label (1 for female, 0 for male). 
The features do not have any particular interpretation. 
Features are continuous values that represent a point in the mdimensional embedding space.
The embeddings have already been computed.
