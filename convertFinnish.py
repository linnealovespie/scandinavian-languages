"""
    Once the raw text files have been generated with `utils.py`, these helper functions create
    word Counters for each decade and saves them as pickled files for easy re-use. 
"""

import utils
import xml.etree.cElementTree  as ET
import os
from bs4 import BeautifulSoup
from io import StringIO
import itertools
from collections import Counter
import pickle

# Finnish Data Values
baseDataPath = "/share/magpie/datasets/Swedish"
earlyPath = os.path.join(baseDataPath, "klk-sv-1771-1879-vrt")
latePath = os.path.join(baseDataPath, "klk-sv-1880-1948-s-vrt")
outputPath = "./Finnish/temp_txt"

# Swedish Data Values
xmlPath = "/home/lm686/data/Swedish/lb/resurser/meningsmangder"
swedishOutputPath = "./Swedish/"

def buildCounter(filePath):
    count = Counter()
    file = os.path.basename(filePath)
    with open(filePath) as reader: 
        words = []
        for line in reader:     
            # If at a text line and not just tags
            if line[0] != "<": 
                wordInfo = line.split()
                #print(wordInfo)
                words += [wordInfo[0]]
                cleanWord = utils.clean_text(wordInfo[0])
                if len(cleanWord) > 0:
                    count[cleanWord] += 1
                #count1880.update(wordInfo[0])

            if len(words) == utils.write_checkpoint:
                words = []

    with open(os.path.join(outputPath, os.path.splitext(file)[0] + ".pickle"), 'wb') as f:
        pickle.dump(count, f) 
        

def buildCounterSwedish(data_dir=xmlPath, dir_out=swedishOutputPath):
    utils.data_dir = data_dir
    utils.dir_out = dir_out
    for xml in os.listdir(data_dir):
        if xml.endswith(".xml"):
            count = Counter()
            print("Parsing Swedish to txt:", xml)
            utils.parse_file_iter(xml, count)
            print("finished counter: ", count.most_common(100))

            with open(os.path.join(dir_out, os.path.splitext(xml)[0] + ".pickle"), 'wb') as f:
                pickle.dump(count, f)
        
if __name__ == "__main__":
    
    buildCounterSwedish()
            