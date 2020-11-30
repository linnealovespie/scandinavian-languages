## Simon Hengchen
# hengchen.net

### IMPORTS

from multiprocessing import Pool 
import os
import operator
from itertools import repeat
import random
import time
import psutil
from tqdm import tqdm
import sys
import platform
import sys
import multiprocessing
import collections
from collections import defaultdict
#import gensim
import itertools
import re
import cProfile, pstats, io

NUM_CORES = multiprocessing.cpu_count()

from utils import *
import xml.etree.ElementTree as ET
import sys


base_dir = os.getcwd()
data_dir = os.path.join("/share/magpie/datasets/Swedish")	# Where the XMLs are
dir_out = os.path.join(base_dir,"/data/Finnish/temp_txt") # Where temp txt will ouput
code_dir = base_dir
earliest_time = 1740

write_checkpoint = 1000000

print("base dir is",base_dir)
print("data dir is",data_dir)
print("output dir is",dir_out)
print("earliest_time is",earliest_time)


if os.path.exists(dir_out) == False:
	os.mkdir(dir_out)
    
def setDirOut(outPath):
    dir_out = outPath
    
def setDirIn(inPath):
    data_dir = inPath

def profile(fnc):
		
	"""A decorator that uses cProfile to profile a function"""
		
	def inner(*args, **kwargs):
			
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print(s.getvalue())
		return retval

	return inner

def update_counter(counter, liste):
    [counter.update(line.split()) for line in liste]
    
#@profile
def parse_file_iter(file, counter):
    """
    This function parses an XML, building the tree iteratively, to extract all words.
    It sends the words to `words_to_lines` every million words, and at the end of the XML.
    """

    file_path = os.path.join(data_dir,file)
    #file_out = os.path.join(dir_out,file.replace(".xml",".txt"))
    #f = open(file_out,"w") ## resetting the original file after testing
    #f.close()

    parser = ET.iterparse(file_path, events=('start','end'))
    parser = iter(parser)
    event, root = next(parser)

    words = []
    for event, elem in parser:
        if elem.tag == "w":
            if event == "end":
                words.append(elem.text)
                elem.clear()
                if len(words) == 1000000:
                    words_to_lines(words,file, counter)
                    words = []
            root.clear()
    words_to_lines(words,file, counter)
    
def words_to_lines(words, file, counter):
    """
    This function transforms a list of tokens into 100-word, space-separated, strings. 
    The words are also cleaned (`clean_text`).
    Every 100,000 strings (= lines) and at the end, the lines are written to disk (`write_to_file`).
    """
    s = ""
    liste = []
    for i, w in enumerate(words):
        if i % 100 != 0:
            s += w.lower() + " "
        else:
            s += w.lower()
            s = clean_text(s)+"\n"
            liste.append(s)
            s = ""
            if len(liste) == 100000:
                update_counter(counter, liste)
                # write_to_file(liste,file)
                liste = []
    update_counter(counter, liste)
    # write_to_file(liste,file)
    

def clean_text(text):
	"""
	Gets a string as input, returns a cleaned out string.
	Cleaning includes removing punctuation, digits, words <= than 2 characters, and non-Swedish characters.
	"""
	texte = re.sub("[0-9]+", " ", text)
	texte = re.sub("[^A-Za-zåäöÖÅÄ\s]", " ", texte)
	texte = " ".join([word for word in texte.split() if not len(word) <= 2])
	texte = " ".join(texte.split())
	return texte

def write_to_file(liste, file):
    file_out = os.path.join(dir_out,file.replace(".vrt",".txt"))
    with open(file_out, "a") as f:
        [f.write(line) for line in liste]

def train_sgns_indep(file,algorithm):

	"""
	This function trains 'independently-trained' embedding models.
	Arguments are: file name and algorithm where 'algorithm' is in ["word2vec", "fasttext"]. 
	If you want to use this code to train embeddings with different hyperparameters, this is where it is done: `model = gensim.models...`
	"""

	dir_out = os.path.join(data_dir,"WE")

	corpus_file = os.path.join(dir_in,file)

	if algorithm == "word2vec":
		model_output = os.path.join(dir_out,"word2vec","indep",file.replace(".gensim",".w2v"))
		if os.path.exists(model_output) == False:
			model = gensim.models.Word2Vec(corpus_file=corpus_file, min_count=50, sg=1 ,size=100, workers=64, seed=1830, iter=5)
			model.save(model_output)
		else:
			print(model_output,"exists")
	
	elif algorithm == "fasttext":
		model_output = os.path.join(dir_out,"fasttext","indep",file.replace(".gensim",".ft"))
		if os.path.exists(model_output) == False:
			model = gensim.models.fasttext.FastText(corpus_file=corpus_file, min_count=50, sg=1 ,size=100, workers=64, seed=1830, iter=5)
			model.save(model_output)
		else:
			print(model_output,"exists")
	
	else:
		print("not a valid algorithm")


def train_sgns_incremental(sorted_list_of_files,algorithm):

	"""
	This function trains 'incrementally-trained' embedding models, following Kim et al 2014.
	Arguments are: SORTED list of filenames, and algorithm where 'algorithm' is in ["word2vec", "fasttext"]. 
	If you want to use this code to train embeddings with different hyperparameters, this is where it is done: `model = gensim.models...`
	"""
	dir_out = os.path.join(data_dir,"WE")
	
	for index, file in enumerate(sorted_list_of_files):
		corpus_file = os.path.join(dir_in,file)
		if index == 0:
			if algorithm == "word2vec":
				model_output = os.path.join(dir_out,"word2vec","incremental",file.replace(".gensim",".w2v"))
				model = gensim.models.Word2Vec(corpus_file=corpus_file, min_count=50, sg=1 ,size=100, workers=64, seed=1830, iter=5)
				model.save(model_output)
			elif algorithm == "fasttext":
				model_output = os.path.join(dir_out,"fasttext","incremental",file.replace(".gensim",".ft"))
				model = gensim.models.fasttext.FastText(corpus_file=corpus_file, min_count=50, sg=1 ,size=100, workers=64, seed=1830, iter=5)
				model.save(model_output)
		elif index > 1:
			if algorithm == "word2vec":
				model_output = os.path.join(dir_out,"word2vec","incremental",file.replace(".gensim",".w2v"))
				model.build_vocab(corpus_file=corpus_file, update=True)
				model.train(corpus_file=corpus_file, total_words = model.corpus_count, total_examples = model.corpus_count, start_alpha = model.alpha, end_alpha = model.min_alpha, epochs=model.epochs)
				model.save(model_output)
			elif algorithm == "fasttext":
				model_output = os.path.join(dir_out,"fasttext","incremental",file.replace(".gensim",".ft"))
				model.build_vocab(corpus_file=corpus_file, update=True)
				model.train(corpus_file=corpus_file, total_words = model.corpus_count, total_examples = model.corpus_count, start_alpha = model.alpha, end_alpha = model.min_alpha, epochs=model.epochs)
				model.save(model_output)