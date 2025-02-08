# Topic Classification of BBC News Articles

## Problem
The goal of this project is to predict the topic of BBC news articles using the Naive Bayes classification algorithm.

## Data Description
- **Training dataset size**: 2125 articles
- **Test dataset size**: 100 articles
- **Columns in both datasets**:
  - `Topic` (category label)
  - `Article` (text content)
- **Topics included**:
  - Sport
  - Tech
  - Business
  - Entertainment
  - Politics

## Training the Model
Training the model involves creating a JSON file that includes:
- Total number of articles
- Unique word count
- Word probabilities per topic

These parameters allow us to calculate the probability of an article belonging to a given topic. The topic with the highest probability is the predicted category.

### Bayes Word Probability Formula:
$P(\text{word} | \text{class}) = \frac{\text{count(word, class)} \ + \ 1}{total words in class \ + \ unique words in all classes}$


## Testing the Model
The trained model is used to predict the topic of test articles by computing the probability of each topic given the words in an article.

### Bayes Probability Formula for Text Classification:
$P(\text{class}) = \log\left( \frac{\text{total texts in class}}{\text{total texts}} \right) + \sum_{i=0}^{n-1} \log \left( P(\text{word}_i | \text{class}) \right)$

Where:
- $\log P(\text{word}_i | \text{class})$ is used if the word appears in the topic.
- $\log(\frac{1}{\text{total words in class} \ + \ \text{total unique words}})$ is used if the word does not appear in the topic.

## Results
The trained model achieved **97% accuracy** in predicting the topic of BBC news articles.


## Conclusion
The Naive Bayes classifier demonstrated high accuracy (97%) in predicting the topics of BBC news articles.
This result highlights its effectiveness in text classification tasks, especially when dealing with well-defined categories and a sufficiently large training dataset.
