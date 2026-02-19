import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import kagglehub
import os
import pandas as pd
import joblib as jbl

wnl = WordNetLemmatizer()
stopwords = stopwords.words("english")

#pattern to match different urls, or hashtags or mentions, or numbers, or punctuation or underscores
garbage_pattern = re.compile(r'(https?://\S+|https?\s\S*|www\.\S+|\bhttps?\b)|[@#]\w+|\d+|[^\w\s]|_')


def load_dataset(dir: str = None):
    try:
        dataset = jbl.load(dir)
    except:

        path = kagglehub.dataset_download("sidharth178/disaster-response-messages")
        datasets = os.listdir(path)

        categories = pd.read_csv(os.path.join(path, datasets[0]))
        messages = pd.read_csv(os.path.join(path, datasets[1]))

        labels = [label.split("-")[0] for label in categories["categories"][0].split(";")]
        categories[labels] = categories["categories"].apply(lambda x: pd.Series(int(label.split("-")[1]) for label in x.split(";")))
        categories.drop("categories", inplace=True, axis=1)
        categories.drop("child_alone", inplace=True, axis=1)

        messages.drop_duplicates(inplace=True)
        categories.drop_duplicates(inplace=True)
        categories["related"] = categories["related"].replace(2, 1)

        dataset = messages.merge(categories, on ="id", how = "inner")
        dataset.drop("original", axis=1, inplace=True)
        if dir != None:
            os.makedirs(os.path.dirname(dir), exist_ok=True)
            jbl.dump(dataset, dir)

    label_cols = dataset.columns[3:]
    return dataset, label_cols

def text_preprocessor(s: str, lemmatize = True, remove_stopwords = True, join = True):
    s = s.lower()              
    #replaces matching stuff with whitespace                 
    s = garbage_pattern.sub(' ', s)
    #cllapses whitespaces and removes trailing, starting whitespaces                
    s = re.sub(r'\s+', ' ', s).strip() 
    s = word_tokenize(s)
    if remove_stopwords:
        s = [w for w in s if w not in stopwords]
    if lemmatize:
        s = [wnl.lemmatize(w) for w in s]
    if join:
        s = ' '.join(s)
    return s