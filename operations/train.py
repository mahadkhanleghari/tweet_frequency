from helpers.clean import cleanhtml, filter_stopwords
import pandas as pd
import structlog
import spacy
from bson.objectid import ObjectId
from collections import Counter
import json
from tqdm import tqdm

log = structlog.get_logger()

class TweetsFrequency:

    def __init__(self, df):
        self.list_dict = ([df.to_dict(orient='records')][0])
        self.nlp = spacy.load("en_core_web_sm")

    def __get_text(self, doc):
        result = [token.text for token in doc]
        return result

    def __pre_process(self):
        log.info("Pre processing data.")
        modified_list, refined_list = [filter_stopwords(self.nlp(cleanhtml(k["tweet"]))) for k in tqdm(self.list_dict)], []
        for tweet in modified_list:
            refined_list.extend(self.__get_text(tweet))
        return dict(Counter(refined_list))

    def __post_process(self, pre_processed):
        log.info("Post processing data")
        main_result = []
        for k in tqdm(pre_processed):
            main_result.append({"id": str(ObjectId()), "word": k, "frequency": pre_processed[k]})
        return main_result

    def compute(self):
        log.info("Computation started.")
        pre_processed = self.__pre_process()
        post_processed = self.__post_process(pre_processed)
        print(json.dumps(post_processed, indent=4))
        return post_processed

if __name__ == "__main__":
    df = pd.read_csv("data/train_tweets.csv")
    tf = TweetsFrequency(df=df)
    result = pd.DataFrame(tf.compute())

    result.to_csv("test_output/test.csv")

