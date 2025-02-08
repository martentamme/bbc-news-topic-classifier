import math
import string

import pandas as pd

from file_helper import FileHelper


class ModelTesting:
    _BBC_TEST_FILENAME = "bbc_test"
    _DATA_COLUMN_NAMES = ["topic", "news_text"]

    def testing(self):
        """
        Bayes probability formula for text classification
        P(w|t) = log(Nt / N) + log(P(w|t))
        """
        df: pd.DataFrame = FileHelper.get_csv_as_df(file_name=self._BBC_TEST_FILENAME,
                                                    column_names=self._DATA_COLUMN_NAMES)
        rows = df.to_dict(orient='records')
        total_rows_count = len(rows)
        model = FileHelper.read_model("model")

        predictions = []
        for row in rows:
            actual_topic = row.get("topic")
            news_text = row.get("news_text")
            words = self._get_words_from_text(news_text)
            topic_probabilities = dict()

            for topic in model["topic_articles_stats"].keys():
                probability = self._topic_probability(model, topic, words)
                topic_probabilities[topic] = probability

            predicted_topic = max(topic_probabilities, key=topic_probabilities.get)
            prediction_stats = dict(
                actual_topic=actual_topic,
                predicted_topic=predicted_topic,
                probabilities=topic_probabilities,
            )
            predictions.append(prediction_stats)

        self._get_model_accuracy(predictions, total_rows_count)
        self._show_predictions_stats(predictions)

    @staticmethod
    def _topic_probability(model: dict, topic: str, words: list[str]) -> float:
        # Base probability
        probability = math.log(model["topic_articles_stats"][topic]["articles_count"] / model["articles_total_count"])

        # Add probability by each word
        for word in words:
            model_words = model["topic_articles_stats"][topic]["words"]
            if model_words.get(word):
                probability += model["topic_articles_stats"][topic]["words"][word]
            else:
                words_count = model["topic_articles_stats"][topic]["words_count"]
                unique_words = model["unique_words_count"]
                if words_count == 0:
                    probability += math.log(1 / unique_words)
                else:
                    probability += math.log(1 / (words_count + unique_words))

        return probability

    @staticmethod
    def _get_model_accuracy(predictions: list, total_rows_count: int):
        predicted_right = 0
        for prediction in predictions:
            if prediction["predicted_topic"] == prediction["actual_topic"]:
                predicted_right += 1

        print(str(predicted_right / total_rows_count * 100) + "% correct")

    @staticmethod
    def _show_predictions_stats(predictions: list):
        for prediction in predictions:
            actual_topic = prediction["actual_topic"]
            predicted_topic = prediction["predicted_topic"]
            print(f"A: {actual_topic} == P: {predicted_topic} -- {prediction['probabilities']}")

    @staticmethod
    def _get_words_from_text(text) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()

        return words


if __name__ == '__main__':
    model_testing = ModelTesting()
    model_testing.testing()
