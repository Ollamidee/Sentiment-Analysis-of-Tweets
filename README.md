# Sentiment Analysis of Twitter Data

### Project Description

This project demonstrates the development of a machine learning model to classify the emotional tone of a given tweet. The goal was to solve a real-world problem of analyzing large-scale text data. The primary challenge was working with a complex dataset containing over 1.6 million tweets; this specific subset, with 40,000 tweets and 13 fine-grained emotion labels, suffered from severe class imbalance.

Through a series of strategic data science techniques, I successfully transformed a model with poor initial performance into a reliable and effective classifier, showcasing a complete and iterative problem-solving workflow.

### Key Features

  * **Data Preprocessing and NLP:** Clean and normalize raw tweet text using NLTK for stop word removal and stemming.
  * **Feature Engineering:** Transform text data into numerical features using the powerful TF-IDF (Term Frequency-Inverse Document Frequency) method.
  * **Model Building:** Train a robust `LogisticRegression` classifier on the processed data.
  * **Problem-Solving:** Implement a custom class aggregation strategy to address class imbalance and high dimensionality.
  * **Performance Evaluation:** Analyze and visualize model performance using a detailed classification report and a bar chart.

### Data

The dataset used for this project is the "Emotion Detection from Text" dataset, publicly available on Kaggle.

  * **Dataset Link:** [https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text?select=tweet\_emotions.csv](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text?select=tweet_emotions.csv)

### Technologies Used

  * **Programming Language:** Python
  * **Libraries:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`
  * **Environment:** Jupyter Notebook

### Installation

To run this project, you need to have Python and the necessary libraries installed.

1.  **Clone the repository:**
    ```
     git clone https://github.com/Ollamidee/Sentiment-Analysis-of-Tweets.git
    ```
2.  **Navigate to the project directory:**
    ```
    cd Sentiment-Analysis-of-Tweets
    ```
3.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```


### How to Use

1.  **Download the dataset** from the link provided above and place the `tweet_emotions.csv` file in a `data/` folder within your project directory.
2.  **Open the Jupyter Notebook:**
    ```
    jupyter notebook
    ```
3.  **Run the cells:** Open the `unt.ipynb` notebook and run each cell sequentially. The notebook is structured to guide you through the data processing, model training, and evaluation steps.

### Project Structure

```
Sentiment-Analysis-of-Tweets/
├── data/
│   └── tweet_emotions.csv
├── sentiment_analysis_of_twitter_data.ipynb
├── requirements.txt
└── README.md
```

### Challenges & Learnings

This project was a valuable lesson in the importance of strategic problem-solving over brute force.

  * **Challenge:** The initial model, trained on raw data, achieved an accuracy of only **32%**. I quickly identified that this was due to the large number of target classes (13 emotions) and the severe class imbalance in the dataset.
  * **Solution:** I chose not to rely on just oversampling techniques like SMOTE, which did not improve performance. Instead, I implemented a custom solution by **aggregating the 13 fine-grained emotions into three broader categories: `positive`, `negative`, and `ambiguous`**. This simplified the problem and was the most impactful step of the project. I also corrected for a small training set by using a standard 80/20 split and resolved a `ConvergenceWarning` by increasing the model's `max_iter` parameter.
  * **Key Learnings:** This process taught me that thoughtful data preparation and feature engineering are often more crucial than the choice of model. A smaller, more logical set of classes can lead to far better results. It also reinforced the importance of adhering to best practices to create a clean, reproducible, and professional project.