import argparse
import pandas as pd
from modules.document import Document
from modules.LDA import LDA
from modules.extract_filetag import get_est_type
from gensim.test.utils import datapath
import gensim.corpora as corpora
from gensim import models


est_type = get_est_type()
csv_filename = "data_" + est_type + ".csv"

verbose = False

def display(text):
    if not verbose:
        return
    print(text)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--data', help='JSON data file', required=False, default=csv_filename)     
    parser.add_argument('-v', help='Verbose', required=False, action='store_true')
    parser.add_argument('-w', help='Save wordcloud image as establisment_lang_wordcloud.png', required=False, action='store_true')
    args = vars(parser.parse_args())

    # Verbose
    if args['v']:
        verbose = True
    
    data = pd.read_csv(args['data'])
    # Remove empty lines
    data.dropna(inplace=True)
    display(data.head())

    # The file tag is get from the first record
    # We beleive that's the same for all records
    #file_tag = data.iloc[0]['est_type'] + "_" +  data.iloc[0]['language']
    # Use python file open() to extract name directly from the file next time. 

    #file_tag = "ASC" + "_" +  data.iloc[0]['language']
    #est_type = get_est_type()

    file_tag = est_type + "_" +  data.iloc[0]['lang']
    
    document = Document()

    # Save word cloud image
    if args['w']:
        document.show_wordcloud(data['clean_text'], file_tag)
    
    # Prepare LDA model by tokenizing text and creating dictionnary
    tokenized_data = document.tokenize(data['clean_text'])
    dictionary = document.create_dictionnary(tokenized_data)

    # Save the dictionary.
    dict_name = 'saved_dicts/dictionary_' + est_type + '.txt'
    dictionary.save(dict_name)
    corpus = document.create_corpus(tokenized_data, dictionary)



    display(f"{len(dictionary)} mots uniques dans {len(corpus)} fragments")

    # LDA
    lda = LDA(dictionary, corpus, file_tag)

    # Search for best coherence score
    lda.coherence(tokenized_data, start=5, max=20, step=2)
