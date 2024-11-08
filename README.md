# Sentiment Analysis of Product Reviews

This project applies natural language processing (NLP) and machine learning to analyze the sentiment of 15,000 product reviews from an online shop. Each review is labeled as either positive or negative. Using this dataset, the project builds a machine learning model that can classify the sentiment of new reviews.

## Project Structure

1. **Loading and Exploring the Dataset**: Load the dataset and examine its structure.
2. **Data Cleaning and Preprocessing**: Clean and preprocess the text data, including removing punctuation, converting text to lowercase, and removing stopwords.
3. **Exploratory Data Analysis (EDA)**: Visualize the distribution of positive and negative reviews, and identify commonly used words.
4. **Preprocessing and Data Splitting**: Tokenize and pad sequences, and split the dataset into training, validation, and test sets.
5. **Model Building**: Construct a neural network with embedding and LSTM layers using TensorFlow.
6. **Model Compilation and Training**: Compile the model with a binary cross-entropy loss function and train it on the data.
7. **Model Evaluation**: Evaluate the model's performance on the test set and display metrics.
8. **Confusion Matrix**: Plot a confusion matrix to visualize classification accuracy.
9. **Predicting Sentiment for New Text**: Test the model on new reviews to verify its predictive capabilities.
10. **Model Saving and Conclusion**: Save the trained model for future use and summarize findings.

## Requirements

- Python 3.x
- Jupyter Notebook
- Required packages:
  - `pandas`
  - `nltk`
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `plotly`

## Usage

- Open the `sentiment_analysis.ipynb` notebook.
- Follow each section to load, preprocess, and analyze the data.
- Train the LSTM model on the dataset and evaluate it to check for accuracy.
- Use the model to predict the sentiment of new reviews.

## Example

To predict the sentiment of a new review:
```python
test_sentence = "This product is fantastic! I love it."
prediction = predict_sentiment(test_sentence)
print("Positive" if prediction > 0.5 else "Negative")
```

## Results

The trained model achieved approximately **82% accuracy** on the test set. It effectively distinguishes positive from negative reviews and can generalize to new data, given that the input is preprocessed similarly.

## Files

- `sentiment_analysis.ipynb`: Main notebook for training and evaluating the model.
- `requirements.txt`: List of required Python packages.
- `sentiment_analysis.pkl`: Serialized model file (created after training).
- `README.md`: Project overview and usage instructions.

## Conclusion

This sentiment analysis model serves as a practical example of using NLP and deep learning for text classification tasks. The results demonstrate that an LSTM network can effectively capture patterns in text data, making it a valuable tool for customer feedback analysis in online retail and other fields.

## Future Improvements

- **Advanced Models**: Implement transformers (e.g., BERT) for potentially higher accuracy.
- **Data Augmentation**: Increase dataset size by augmenting text data for more robust training.
