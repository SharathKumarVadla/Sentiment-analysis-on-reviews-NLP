# SENTIMENT ANALYSIS ON AMAZON FINE FOOD REVIEWS

**Objective** - The objective of this project is to build a robust and scalable solution for binary sentiment classification on the Amazon Fine Food Reviews dataset. The goal is to predict whether a given review reflects a positive or negative sentiment based solely on the textual content of the review.

**Background and Motivation**:
Customer reviews are a valuable source of feedback for businesses, helping them understand consumer preferences, identify areas of improvement, and optimize their products or services. The Amazon Fine Food Reviews dataset, which contains over half a million food product reviews, provides an excellent opportunity to analyze sentiment patterns, explore natural language processing (NLP) techniques, and apply deep learning models for practical insights.

This project focuses on converting raw, unstructured textual data into actionable insights by:

* Cleaning and preprocessing noisy review text.
* Building a predictive model that can classify reviews into positive or negative categories.
* Understanding the impact of preprocessing techniques on model performance.

**Use Case Scenarios**:

**1. Business Decision Support:**

By analyzing customer feedback, businesses can automatically flag negative reviews for quicker resolution, improving customer satisfaction and brand loyalty.
Even without product category information, this insight remains valuable for addressing general customer concerns.

**2. Sentiment Trends Across Reviews:**

Aggregating sentiment scores over time helps businesses identify shifts in customer perceptions and evaluate the impact of campaigns or product changes.

**Dataset**

The Amazon Fine Food Reviews dataset consists of over 500,000 customer reviews for various food products available on Amazon. Each review includes multiple attributes such as the review text, numerical score (rating), and metadata like product and user IDs. The dataset captures customers' opinions and experiences, providing a rich resource for sentiment analysis and natural language processing tasks.

The key columns used in this project are:

Text  - The full review content provided by customers.<br>
Score - The numerical rating assigned by the customer, ranging from 1 to 5, which reflects their sentiment towards the product.

Link - *https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews*

**Subsampling for Efficiency:**

Inorder to balance computational requirements with model performance, I have randomly selected a manageable subset of the dataset (e.g., 100,000 reviews).

**Sentiment Transformation:**

I have converted review scores into binary labels as below.

* Positive Sentiment (1): Reviews with scores greater than 3.
* Negative Sentiment (0): Reviews with scores less than or equal to 2.
* Exclude neutral reviews (score == 3) to maintain a clear distinction in the classification task.

**Data Processing:**

- Stripping HTML tags to ensure the text contains only meaningful content without markup elements.  
- Expanding contractions to their full forms, standardizing the text for consistent word representations.  
- Retaining only alphabetic words by removing numbers, special characters, and punctuation, reducing noise in the data.  
- Converting all text to lowercase to ensure consistency, treating words like *Food* and *food* as the same.  
- Removing common stopwords like *the*, *and*, and *is* focusing on words that contribute to sentiment.  
- Returning the cleaned and normalized text, making it ready for feature extraction and analysis.  

**Featurization:**

The project experiments with two feature extraction techniques for sentiment analysis: **TF-IDF** and **BERT vector representations**. The BERT model, consisting of 109,482,241 parameters, is used without fine-tuning. Instead, it provides pre-trained vector representations, which are used as input features for the classification model.

The extracted features are fed into a fully connected neural network with the following architecture:
- **Input Layer**: Accepts feature vectors of size equal to the output dimensions of the feature extraction technique.
- **Dense Layers**: Five hidden layers with progressively decreasing units (128, 64, 32, 16, and 8) and ReLU activation functions. These layers capture non-linear patterns in the input data.
- **Output Layer**: A dense layer with 2 units and a softmax activation function to classify text as either positive or negative sentiment.

The model is compiled using the **Adam optimizer** with a learning rate of 0.001 and categorical crossentropy as the loss function. The **F1-score** is used as an evaluation metric to balance precision and recall. This architecture is designed to leverage both traditional (TF-IDF) and modern (BERT) vector representations for effective sentiment classification.

**Results:**

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/3d6da8b7-8885-4fa7-96de-dfbcdc87cbeb" alt="Image 1" style="width: 48%; margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/508db8ff-308d-465c-baf1-6eb7779cc09d" alt="Image 2" style="width: 48%;">
</div>

$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ *Fig - Plots for TFIDF*  $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ *Fig - Plots for BERT*

<div align="center">
  <table>
    <tr>
      <th>Model</th>
      <th>Dataset</th>
      <th>Loss</th>
      <th>F1 Score</th>
    </tr>
    <tr>
      <td>BERT</td>
      <td>Train</td>
      <td>0.2900</td>
      <td>0.9311</td>
    </tr>
    <tr>
      <td>BERT</td>
      <td>Test</td>
      <td>0.2932</td>
      <td>0.9319</td>
    </tr>
    <tr>
      <td>TF-IDF</td>
      <td>Train</td>
      <td>0.1779</td>
      <td>0.9610</td>
    </tr>
    <tr>
      <td>TF-IDF</td>
      <td>Test</td>
      <td>0.2371</td>
      <td>0.9465</td>
    </tr>
  </table>
</div>

The TF-IDF model performs better overall. It has a higher F1 score during training (0.9610 compared to BERT's 0.9311) and performs well on the test data as well (0.9465 compared to BERT's 0.9319). Even though both models have low loss, TF-IDF demonstrates stronger performance, particularly in training.

