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




