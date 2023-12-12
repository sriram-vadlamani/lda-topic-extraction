from tqdm import tqdm
from wordcloud import WordCloud
import gensim.corpora as corpora
import numpy as np

class Document:
    def tokenize(self, sentences):
        """Tokenzise text by splitting it by space"""  
        tokenized_data = [] 
        for sentence in list(sentences):
            tokenized_data.append(sentence.split(" "))
            
        return tokenized_data

    def create_dictionnary(self, tokenized_data):
        """Create dictionary from dataset"""
        return corpora.Dictionary(tokenized_data)

    def create_corpus(self, tokenized_data, dictionary):
        return [dictionary.doc2bow(line) for line in tokenized_data] 

    def show_wordcloud(self, data, file_tag):
        # Display Word cloud
        all_words = ' '.join(list(data))
        wordcloud = WordCloud(
            background_color="white", 
            max_words=5000, 
            contour_width=3, 
            contour_color='steelblue',
            width=800,
            height=500)
        wordcloud.generate(all_words)
        image = wordcloud.to_image()
        image.save("data/" + file_tag + "_wordcloud.png")
