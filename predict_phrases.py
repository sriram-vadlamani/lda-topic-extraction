import gensim
import argparse
from gensim import models
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
from gensim.test.utils import datapath
from modules.extract_filetag import get_est_type

verbose = False

est_type = get_est_type()
csv_filename = "data_" + est_type + ".csv"

def display(text):
    if not verbose:
        return
    print(text)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-v', help='Verbose', required=False, action='store_true')
    parser.add_argument('-t', '--topics', help='Number of topics', required=True)
    args = vars(parser.parse_args())


    if args['v']:
        verbose = True


    # Load the saved model
    lda_name = 'saved_models/lda_' + est_type + '.model'
    loaded_lda = gensim.models.LdaModel.load(lda_name)

    num_topics = int(args['topics'])
    
    topics = {}
    for topic in loaded_lda.print_topics(num_topics, 5):
    	topics[topic[0]] = topic[1]

    dictionary_name = 'saved_dicts/dictionary_' + est_type + '.txt'
    dictionary = corpora.Dictionary.load(dictionary_name)
    df = pd.read_csv(csv_filename)
    df_new = df[df['clean_text'].notnull()]
    for index, row in df_new.iterrows():
    	new_vec = dictionary.doc2bow(row['clean_text'].split(" "))
    	pred = loaded_lda[new_vec]
    	topic = max(pred, key=lambda item: item[1])[0]
    	print(f"Topic {topic} - {topics[topic]}")
    	print(f"{row['content']}")
