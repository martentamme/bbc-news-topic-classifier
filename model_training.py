import math

import pandas as pd

from file_helper import FileHelper

import string

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


class ModelTraining:
    _BBC_TRAIN_FILENAME = "bbc_train"
    _MODEL_PATH = "/model"
    _DATA_COLUMN_NAMES = ["topic", "news_text"]

    def __init__(self):
        # setup
        self.topic_articles_stats = {}
        self.articles_total_count = 0
        self.unique_words = set()
        self.stop_words = set(stopwords.words('english'))

    def build_model(self):
        self._prepare_training_stats()
        total_stats = self._get_total_stats()
        topics = total_stats["topic_articles_stats"].keys()

        for topic in topics:

            # Modify words count to probability
            for word in total_stats["topic_articles_stats"][topic]["words"].keys():
                words_count = total_stats["topic_articles_stats"][topic]["words_count"]
                # Bayes formula
                # P(word | class) = (count(word, class) + 1) / (total words in class + unique words in all classes)
                probability = math.log((total_stats["topic_articles_stats"][topic]["words"][word] + 1) / (
                            words_count + total_stats['unique_words_count']))
                total_stats["topic_articles_stats"][topic]["words"][word] = probability

        FileHelper.write_model("model", total_stats)

    def _prepare_training_stats(self):
        df: pd.DataFrame = FileHelper.get_csv_as_df(file_name=self._BBC_TRAIN_FILENAME,
                                                    column_names=self._DATA_COLUMN_NAMES)
        rows = df.to_dict(orient='records')
        self.articles_total_count = len(rows)

        for i, row in enumerate(rows):
            print(i + 1)
            topic = row.get("topic")
            news_text = row.get("news_text")

            # Add new topic
            if topic not in self.topic_articles_stats:
                self.topic_articles_stats[topic] = {"articles_count": 0, "words_count": 0, "words": {}}

            # Increase articles count
            self.topic_articles_stats[topic]["articles_count"] = self.topic_articles_stats[topic]["articles_count"] + 1

            # Increase words count
            words: list[str] = self._get_words_from_text(news_text)
            self.topic_articles_stats[topic]["words_count"] = self.topic_articles_stats[topic]["words_count"] + len(
                words)

            # Extend words stats
            self._extend_words_stats(topic, words)

    def _extend_words_stats(self, topic: str, words: list[str]) -> None:
        """
        Loop through words and:
        1. Update words dictionary {word: count}
        2. Update unique words set
        """
        for word in words:
            dictionary = self.topic_articles_stats[topic]

            # Update words dictionary
            if not (word in dictionary["words"]):
                dictionary["words"][word] = 0

            dictionary["words"][word] = dictionary["words"][word] + 1

            # Update unique words
            self.unique_words.add(word)

    def _get_total_stats(self):
        return dict(
            articles_total_count=self.articles_total_count,
            unique_words_count=len(self.unique_words),
            topic_articles_stats=self.topic_articles_stats
        )

    def _get_words_from_text(self, text) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in self.stop_words]

        return words


if __name__ == '__main__':
    model_training = ModelTraining()
    model_training.build_model()
