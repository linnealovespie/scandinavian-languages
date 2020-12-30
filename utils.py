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
from collections import Counter
import pickle
import numpy as np

NUM_CORES = multiprocessing.cpu_count()

from utils import *
import xml.etree.ElementTree as ET
import sys


base_dir = os.getcwd()
data_dir = os.path.join("/share/magpie/datasets/Swedish")	# Where the XMLs are
dir_out = os.path.join(base_dir,"data/Finnish/temp_txt") # Where temp txt will ouput
code_dir = base_dir
earliest_time = 1740

write_checkpoint = 1000000

print("base dir is",base_dir)
print("data dir is",data_dir)
print("output dir is",dir_out)
print("earliest_time is",earliest_time)


class LanguageCounter():
    """ Class used to represent each corpus. 
    ...
    Attributes
    ----------
    dataPath: str
        Path to the directory that contains the pickled raw text for each newspaper. 
    allCounters: Dict[np.datetime64, Counter]
        A mapping from decade to a counter of each word in the newspaper publishings for that decade. 
    commonWords: Dict[np.datetime64, list[str]]
        A mapping from decade to the top 100 most commonly used words used in that decade. 
    topWordsTotal: Counter
        The top 250 words used across all time by the newspapers and their overall frequncies. 
    allFeatized: List[List[float]]
        A list of 250-dimensional vectors, one for each decade of the corpus. Vectors show the frequencies of each of the topWordsTotal within that decade.
    
    Methods
    -------
    buildCommonCounters(dataPath)
        For every pickle file in dataPath, adds a counter for the given decade to self.allCounters
    getOverlaps()
        Get the amount of word overlap across decades. 
    buildFeatures()
        Builds feature vectors that describe each decade by the frequency of each of the overall top 250 most commonly used words by the language. 
    """
    
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.allCounters = {} 
        self.commonWords = {}
        
        self.topWordsTotal = None
        self.allFeatized = None
        
        #buildCommonCounters(self.datapath)
        #buildFeatures()
    
    """
        For every pickle file in dataPath, adds a counter for each decade to self.allCounters
    """
    def buildCommonCounters(self, dataPath):
        # commonWords = []
        # allCounters = {d: Counter() for d in decades}
        # TODO: re-run for finnish 1771 bc im dumb
        for f in os.listdir(dataPath):
            if(f.endswith(".pickle")):
                decade = os.path.basename(f)
                decade = decade[decade.index("1"): decade.index("1")+3] + "0"
                decade = np.datetime64(decade, 'Y')
                if decade not in list(self.allCounters.keys()):
                    # print("adding decade", decade)
                    self.allCounters[decade] = Counter()
                    self.commonWords[decade] = set([])

                with open(os.path.join(dataPath, f), 'rb') as f:
                    wordCounter = pickle.load(f)

                self.allCounters[decade] += wordCounter
                top100 = [w[0] for w in wordCounter.most_common(100)]
                # Build a list of the top 100 across all files of a decade
                # Mainly helps get an idea over anything quantitative
                self.commonWords[decade].update(top100)
        self.allCounters = {k: self.allCounters[k] for k in sorted(self.allCounters)}
        self.commonWords = {k: self.commonWords[k] for k in sorted(self.commonWords)}
    
    def getOverlaps(self):
        overlaps = []
        simToBase = [1]
        base = list(self.commonWords.keys())[0]
        for i in range(len(self.commonWords)-1):
            decade = list(self.commonWords.keys())[i]
            decade_n = list(self.commonWords.keys())[i+1]
            overlaps += [len(set(self.commonWords[decade]).intersection(self.commonWords[decade_n]))]
            simToBase += [len(set(self.commonWords[decade_n]).intersection(self.commonWords[base]))
                         / len(self.commonWords[decade_n].union(self.commonWords[base]))]
        # overlapBounds = len(set(self.commonWords[0]).intersection(self.commonWords[-1]))
        return overlaps, simToBase
    
    def buildFeatures(self):
        totalFreqs = sum(self.allCounters.values(), Counter())
        totalFreqsSorted = totalFreqs.most_common()
        self.topWordsCounter = totalFreqs.most_common(250)
        
        self.topWordsTotal = []
        i = 0
        while len(self.topWordsTotal) < 250: 
            word = totalFreqsSorted[i]
            inText = [1 if word[0] in list(self.allCounters.values())[j] else 0 for j in range(len(self.allCounters))]
            proport = sum(inText) / len(inText)
            print("word ", word, "in % of texts: ", proport)
            if proport >= 0.5: 
                self.topWordsTotal += [word[0]]
            i += 1
            
        self.allFeatize = []
        for counter in list(self.allCounters.values()):
            lenDoc = sum(counter.values())
            featize = np.array([counter[self.topWordsTotal[i]] for i in range(len(self.topWordsTotal))])
            featize = np.divide(featize, lenDoc)
            self.allFeatize += [featize]


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