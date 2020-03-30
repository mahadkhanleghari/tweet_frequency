import re
import spacy

nlp = spacy.load("en_core_web_sm")

def cleanhtml(raw_html):
    cleanr = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    cleantext = re.sub(cleanr, "", raw_html)
    return cleantext

def filter_stopwords(doc):
    token_list = [token.text for token in doc]
    filtered_sentence = []
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop is False:
            filtered_sentence.append(word)
    refined_doc = nlp(" ".join(filtered_sentence))
    return refined_doc
