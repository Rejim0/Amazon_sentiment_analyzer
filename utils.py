import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



lemm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))- {"not", "no", "never", "neither", "nor",
"hardly", "barely", "scarcely", "n't"}

contraction_fragments = {"wo", "ca", "sha","br"}

stop_words = stop_words | contraction_fragments
#%%
def text_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalpha() or i == "n't":
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        lem_words = lemm.lemmatize(i)
        y.append(lem_words)

    return " ".join(y)