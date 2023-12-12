import unicodedata
import string
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stopwords

class Clean:

    def __init__(self, lang):
        self.lang = lang        
        self.nlp = spacy.load('fr_dep_news_trf')        
        self.stopwords = fr_stopwords

    def mr_proper(self, data):

        # Lower case
        data = data.lower()

        # lemmatizer
        remove_tag = ["DET","CCONJ","AUX","SCONJ","PRON"]
        doc = self.nlp(data)
        lemmatized = []
        for token in doc:
          if token.tag_ in remove_tag:
              continue
          lemmatized.append(token.lemma_)
        data = " ".join(lemmatized)

        # Remove accents
        data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf_8','ignore')

        # Remove characters with digits
        table_ = str.maketrans('', '', string.digits)
        data = data.translate(table_)

        # Replace punctuation by white space
        translator = str.maketrans(
            string.punctuation, ' ' * len(string.punctuation))
        data = data.translate(translator)
       
        # Stop words
        data = [x for x in data.split(' ') if x not in self.stopwords]
        data =list(filter(None, data))
        data = " ".join(data)

        # Extra stop words
        stop = ["avoir", "a", "tres", "tre", "dire", "plu","bon","tout","meme"]
        data = [x for x in data.split(' ') if x not in stop]
        data =list(filter(None, data))
        data = " ".join(data)

        # Only word this more than 3 letters
        data = [x for x in data.split(' ') if len(x) > 3]
        data =list(filter(None, data))
        data = " ".join(data)
        
        return data