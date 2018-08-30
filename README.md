###############################

This is an adapted version of the code used in:
Hill, F., Cho, K., Korhonen, A., & Bengio, Y. (2015). Learning to understand phrases by embedding the dictionary. arXiv preprint arXiv:1504.00548 

Thanks to Dr. Hill for providing the original code.

Please also note that included in this repository is the entirety of the subword-nmt repository, credit: 
Rico Sennrich, Barry Haddow and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

The original repo can be found here: 
https://github.com/rsennrich/subword-nmt

###############################

### Neural Definition Model ###

This repository contains the key parts of code used for my MSc Dissertation project (University of Edinburgh).

The code is written in Python using Tensorflow. For resonable training times, the models must be trained using a GPU. To run models on a CPU, or to test an implementation, the batch size can be set to 128 and embedding size can be set to a small number (10) for very fast training times. Note that the results will not be comparable to a fully trained model.

It is a Recurrent Neural Network Definition Model. Similar to a language model, a definition model takes an input and outputs a single word. The difference being that the definition model does not intend to predict the next word in a sequence, rather the word that is being described by the input definition. 

In order to get a prediction, the definition is encoded into a single embedding, which is then compared (using cosine similarity) with a vocabulary of pretrained embeddings to find the closest embedding which is the output word.

The model uses LSTMs to encode the sequence. I have extended the code with functionality to use either a bidirectional LSTM or to take the average of LSTM states over the sequence (LSTM average). 

The training objective is either cosine loss or rank loss. Rank loss was not included in the original implementation and the model was extended with this functionality. 

For my dissertation project, a number of experiments were carried out using sub-word units. The algorithm used is byte-pair encoding (BPE) as described in Sennrich et al. 2016, cited above. It is also possible to conduct BPE experiments using this code, described below.


### Using the code ###

In order to use the code, the datasets and word embeddings must be downloaded from a separate source (due to size constraints on GitHub). Word embeddings are a reduced version of the GloVe 6B dataset, available from: https://nlp.stanford.edu/projects/glove/.

The original training set is taken from Hill et al (2015), cited above. It contains ~850k word/definition pairs extracted from several dictionaries plus wikipedia.
The full training set also includes ~300k general knowledge crossword questions taken from the New York Times, and ~100k crossword questions taken from The Guardian. These were gathered from the HTML page code, and are strictly for academic use. 
Disclaimer: I do not own the rights to these nor am I attempting to publish them for any other use. 

Here is a link to a Google Drive contianing the datasets: 
https://drive.google.com/drive/folders/1eWzYmY9UhLDuvfUHzMh9NkqYmnqZ4PWv?usp=sharing

These must be downloaded and the data folder must be added into the main repository (at the same level as the Python code files).

For BPE-level experiments, the code to segment the training and test data and apply BPE is included (run_bpe.sh). It will populate the empty BPE folders.

Functionality for adjustable options (bidirectional LSTM / LSTM average, rank loss / cosine loss) is included in the code, and can be selected in the config file, along with the other adjustable hyperparameters. To conduct BPE-level experiments, or to change training sets, the 
config file must be adjusted accordingly. The data directory must be changed to the desired directory, and train, dev and test files must be correct (note that the BPE datasets will have the extension '.bpe' and word-level datasets will be '.tok'). For BPE experiments, the vocab size will also be smaller. The code can be run once with a vocab size of 10k, then reduced according to the error message (this means the system will never have to deal with OOV words on the input side!).

Once the datasets and config file is set up, the code can simply be run from the terminal by entering: python neural_definition_model_full.py

Currently the model only outputs the median rank of the correct word on the test set, which was required for testing the performance of each model. In a future update, the querying function will be restored so that a user can input a definition and see what the model predicts.

Further updates are scheduled including the addition of a decoder to allow for sub-word, multiple-word, and even character-level output.

###############################

If you would like to contact me with any questions or comments about the implementation please email me at Jack_Parry@hotmail.com. 
Any suggestions for improvements etc. are welcomed.


############# END #############
