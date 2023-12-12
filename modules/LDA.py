from tqdm import tqdm
import gensim
from gensim import models
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
from gensim.test.utils import datapath
import modules.extract_filetag as ef


est_type = ef.get_est_type()
csv_filename = "data_" + est_type + ".csv"

class LDA:
  
  def __init__(self, dictionary, corpus, file_tag):
    self.dictionary = dictionary
    self.corpus = corpus
    self.file_tag = file_tag

  def train(self, num_topics):
    """Train LDA model"""
    model = gensim.models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,            
            num_topics=num_topics,
            random_state=63            
          )
    return model

  

  def coherence(self,tokenized_data, start, max, step):
    """Calculate coherence score for each  num ogf topics and display graph"""
    coherence_values = []
    print("| Nb thèmes  | Score de cohérence |")
    print("|------------|--------------------|")
    for num_topics in range(start, max, step):
        model = self.train(num_topics)
        coherence_model = CoherenceModel(model=model, texts=tokenized_data, dictionary=self.dictionary, coherence='c_v')
        coherence_value = coherence_model.get_coherence()
        coherence_values.append(coherence_value)
        print(f"|{num_topics} | {coherence_value:.4f} |")

    x = range(start, max, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig("data/" + self.file_tag + "_coherence.png")
    #plt.show()


  def save_model(self, model):
    """ Save the model for later predictions """
    lda_name = 'saved_models/lda_' + est_type + '.model'
    model.save(lda_name)
  
  def show_topics(self, num_topics=12, num_words=5):  
    """Show first five keyword of each topic""" 
    model = self.train(num_topics)
    self.save_model(model)
    topics = {}
    
    for topic in range(num_topics):
      topics[topic] = [i[0] for i in model.show_topic(topic, topn=num_words)]; 
    return pd.DataFrame(topics)
