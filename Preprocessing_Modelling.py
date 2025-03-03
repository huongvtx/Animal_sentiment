# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:28:50 2024

@author: vutxu
"""

###############################################################################
#-------------------------  Preprocessing section  ---------------------------#
###############################################################################

import pandas as pd
import re
import emot
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from spelling_corrector import correct_spelling_batch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Import data
file_1 = 'Merged_AnimalData_raw.xlsx'
data = pd.read_excel(file_1, sheet_name='comment')


"""
Remove URLs, hashtags, mentions; Normalise apostrophes; Lowercasing
"""
# Define a function
def rm_url_htg_mtn(text):
    if isinstance(text, str):  # Ensure it's a string before applying regex
        # Convert text to lowercase
        text = text.lower()
        # Remove URLs
        # text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\b(?:http|https|ftp)://\S+|www\.\S+\b', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+(\.\w+)*', '', text)
        # Replace curly apostrophes and backticks with straight ones
        text = text.replace("’", "'").replace("`", "'").replace("‘", "'").replace("´", "'")
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply the function to the column of raw comments and create a new column
data['cleaned_cmt'] = data.iloc[:, 5].apply(rm_url_htg_mtn)


"""
Split raw comment into 3 columns: text, emoji, emoticon
"""
# Initialize emot package
emot_obj = emot.core.emot()

emoji_results = emot_obj.bulk_emoji(data['cleaned_cmt'].tolist())
emoticon_results = emot_obj.bulk_emoticons(data['cleaned_cmt'].tolist())

# Function to remove characters from a string based on locations
def remove_by_location(text, locations):
    if not isinstance(text, str):  # Check if the input is a string
        return ''
    # Convert the string to a list of characters for easier modification
    text_list = list(text)
    
    # Iterate over the locations and remove the corresponding characters
    for start, end in sorted(locations, reverse=True):
        del text_list[start:end]  # Delete characters between start and end positions
    
    # Join the modified list back into a string
    cleaned_text = ''.join(text_list).strip()
    
    # Use regex to remove extra spaces between words
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text

# Clean the text by removing emojis and emoticons based on their locations
data['cmt_text'] = [
    remove_by_location(
        remove_by_location(text, emoji_result['location']),  # First remove emojis
        emoticon_result['location']  # Then remove emoticons
    )
    for text, emoji_result, emoticon_result in zip(data['cleaned_cmt'], emoji_results, emoticon_results)
]

data['cmt_text'] = data['cmt_text'].str.strip()

# Extract emojis and emoticons to separate columns
data['cmt_emoji'] = [' '.join(result['value']) for result in emoji_results]
data['cmt_emoticon'] = [' '.join(result['value']) for result in emoticon_results]


"""
Translate slangs to conventional words
"""
# Load the slang dictionary into a Python dictionary
def load_slang_dictionary(file_path):
    slang_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='=')
        for row in reader:
            slang_dict[row[0].lower()] = row[1]
    return slang_dict

# Translator function using the pre-loaded slang dictionary
def translator(text, slang_dict):
    if isinstance(text, str):
        words = text.split()
        translated_words = []
        for word in words:
            cleaned_word = re.sub('[^a-zA-Z0-9-_]', '', word).lower()
            translation = slang_dict.get(cleaned_word, word)
            # Preserve uppercase
            if word.isupper():
                translation = translation.upper()
            translated_words.append(translation)
        return ' '.join(translated_words)
    return text

# Load the slang dictionary
slang_dict = load_slang_dictionary('slang.txt')

# Apply the translator function only to valid strings, ignore NaN or non-string data
data['cmt_text2'] = data['cmt_text'].apply(lambda x: translator(x, slang_dict).lower())


"""
Handle negation: Convert negation words to "nnot" to keep them from stopwords
"""
# Load the negation dictionary into a Python dictionary
def load_neg_dictionary(file_path):
    neg_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='=')
        for row in reader:
            neg_dict[row[0]] = row[1]
    return neg_dict

# Converter function using the pre-loaded negation dictionary
def converter_neg(text, neg_dict):
    if isinstance(text, str):
        words = text.split()
        converted_words = []
        for word in words:
            cleaned_word = re.sub('[^a-zA-Z0-9-_\']', '', word)
            convertion = neg_dict.get(cleaned_word, word)
            converted_words.append(convertion)
        return ' '.join(converted_words)
    return text

# Load the negation dictionary once
neg_dict = load_neg_dictionary('negation.txt')

# Apply the converter function only to valid strings, ignore NaN or non-string data
data['cmt_text3'] = data['cmt_text2'].apply(lambda x: converter_neg(x, neg_dict))


"""
Remove contractions (characters behind apostrophes) & special characters
"""
# Remove the apostrophe and part after it for each row
data['cmt_text3'] = data['cmt_text3'].apply(lambda x: re.sub(r"'[a-zA-Z]+", '', x) if isinstance(x, str) else x)

# Remove special characters (keeping only alphabetical characters and spaces)
data['cmt_text3'] = data['cmt_text3'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)


"""
Correct spelling ("nnot" inclusive)
"""
def parallel_correct_spelling(data, column_name, num_processes=4):
    texts = data[column_name].tolist()
    chunk_size = len(texts) // num_processes
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    with Pool(num_processes) as pool:
        results = pool.map(correct_spelling_batch, chunks)
    
    corrected_texts = [text for sublist in results for text in sublist]
    return corrected_texts

data['cmt_text4'] = parallel_correct_spelling(data, 'cmt_text3')


"""
Remove short words
"""
# Function to remove words with 1 or 2 characters
def remove_short_words(text):
    # Use regex to remove words that have 1 or 2 characters
    return re.sub(r'\b\w{1,2}\b', '', text)

# Apply the function to the column
data['cmt_text4'] = data['cmt_text4'].apply(lambda x: remove_short_words(x) if isinstance(x, str) else x)

# Strip any extra spaces left after removing short words
data['cmt_text4'] = data['cmt_text4'].str.strip().replace(r'\s+', ' ', regex=True)


"""
Tokenise & Remove stopwords
"""
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
# List of words to exclude
excl_ls = ["not", "nor", "no", "but"]

# Remove the words in excl_ls from the stopwords list
stop_words = stop_words - set(excl_ls)

# Define a function to remove stopwords from text
def rm_stopwords(text):
    if isinstance(text, str):
        # Tokenize the text
        words = word_tokenize(text)  # Convert text to lowercase
        # Filter out stopwords
        filtered_words = [word for word in words if word not in stop_words]
        # Rejoin the filtered words into a string
        return ' '.join(filtered_words)
    return text

# Apply the function to the desired column in the DataFrame
data['cmt_text5'] = data['cmt_text4'].apply(rm_stopwords)



"""
Measure sentiment score of text, emojis, and emoticons
"""
# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to extract emojis/emoticons and compute VADER sentiment score
def extract_emos_and_score(row):
    emojis = row['cmt_emoji'].split() if pd.notna(row['cmt_emoji']) else []
    emoticons = row['cmt_emoticon'].split() if pd.notna(row['cmt_emoticon']) else []
    
    emo_list = emojis + emoticons
    scores = [analyzer.polarity_scores(emo)['compound'] for emo in emo_list]
    
    return pd.DataFrame({'emo': emo_list, 'vader_score': scores})

# Apply the function and concatenate results into a new DataFrame 'emo_df'
emo_df = pd.concat([extract_emos_and_score(row) for _, row in data.iterrows()], ignore_index=True)

# Import the emoji sentiment ranking Excel file
file_2 = 'emo_processing_list.xlsx'
emoji_rank = pd.read_excel(file_2, sheet_name="sent_short")

# Merge emo_df with emoji_rank on the 'emo' column
emo_df = emo_df.merge(emoji_rank, on='emo', how='left')

# If emo_rank is missing, use vader_score; otherwise, calculate the mean of both
emo_df['final_score'] = emo_df.apply(lambda row: (row['emo_rank'] + row['vader_score']) / 2 
                                     if pd.notna(row['emo_rank']) 
                                     else row['vader_score'], axis=1)

# Function to calculate sentiment score for both emojis and emoticons in a row
def cal_emo_sentiscore(emojis, emoticons):
    total_score = 0  # Initialize total score
    count = 0
    
    # Process emojis
    if pd.notna(emojis):
        for emoji in emojis.split():
            score = emo_df.loc[emo_df['emo'] == emoji, 'final_score'].values
            if score.size > 0:
                total_score += score[0]  # Add emoji score to total_score
                count += 1
    
    # Process emoticons
    if pd.notna(emoticons):
        for emoticon in emoticons.split():
            score = emo_df.loc[emo_df['emo'] == emoticon, 'final_score'].values
            if score.size > 0:
                total_score += score[0]  # Add emoticon score to total_score
                count += 1
    
    return total_score / count if count > 0 else None

# Apply the function to calculate cumulative sentiment score for each row
data['emo_senti_score'] = data.apply(lambda row: cal_emo_sentiscore(row['cmt_emoji'], row['cmt_emoticon']), axis=1)

# Function to calculate sentiment score for text only
def cal_text_sentiscore(text):
    # Check if the text is None, not a string, or empty
    if text is None or not isinstance(text, str) or text.strip() == "":
        return None
    else:
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']


# Apply the function to the 'text_only' column
data['text_senti_score'] = data['cmt_text5'].apply(cal_text_sentiscore)

# Function to convert polarity score to sentiment
def cal_senti(score):
    if pd.notna(score):
        if -0.05 <= score <= 0.05:
            return 'neutral'
        elif score > 0.05:
            return 'pos'
        elif score < -0.05:
            return 'neg'

# Apply function to polarity score
data['emo_sentiment'] = data['emo_senti_score'].apply(cal_senti)
data['text_sentiment'] = data['text_senti_score'].apply(cal_senti)


# Function to calculate final sentiment based on text_senti and emo_senti
def calculate_final_sentiment(row):
    text_senti = row['text_sentiment']
    emo_senti = row['emo_sentiment']
    
    # Check for the various conditions
    if pd.isna(text_senti) and pd.isna(emo_senti):
        return None
    
    elif (pd.isna(text_senti) or text_senti == 'neutral') and emo_senti == 'pos':
        return "pos"
    elif (pd.isna(text_senti) or text_senti == 'neutral') and emo_senti == 'neg':
        return "neg"
    elif (pd.isna(text_senti) or text_senti == 'neutral') and emo_senti == 'neutral':
        return "neutral"
    
    elif text_senti == 'pos' and (pd.isna(emo_senti) or emo_senti== 'neutral'):
        return "pos"
    elif text_senti == 'neg' and (pd.isna(emo_senti) or emo_senti== 'neutral'):
        return "neg"
    elif text_senti == 'neutral' and (pd.isna(emo_senti) or emo_senti== 'neutral'):
        return "neutral"
    
    elif text_senti == 'pos' and emo_senti == 'pos':
        return "pos"
    elif text_senti == 'neg' and emo_senti == 'neg':
        return "neg"
    elif text_senti == 'pos' and emo_senti == 'neg':
        return "neutral"
    elif text_senti == 'neg' and emo_senti == 'pos':
        return "neutral"

# Apply the function to create the 'final_senti' column
data['cmt_final_senti'] = data.apply(calculate_final_sentiment, axis=1)


"""
Process the same for posts, but without splitting emojis nor checking spelling
"""
# Create a new DataFrame df_post with unique "post" values
df_post = data[['source', 'animal', 'news media', 'post']].drop_duplicates().reset_index(drop=True)
# Remove irrelevant text
df_post['cleaned_post1'] = df_post['post'].apply(rm_url_htg_mtn)
# Translate slangs
df_post['cleaned_post2'] = df_post['cleaned_post1'].apply(lambda x: translator(x, slang_dict).lower())
# Handle negation
df_post['cleaned_post3'] = df_post['cleaned_post2'].apply(lambda x: converter_neg(x, neg_dict))
# Remove the apostrophe and part after it for each row
df_post['cleaned_post3'] = df_post['cleaned_post3'].apply(lambda x: re.sub(r"'[a-zA-Z]+", '', x) if isinstance(x, str) else x)
# Remove special characters (keeping only alphabetical characters and spaces)
df_post['cleaned_post3'] = df_post['cleaned_post3'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x) if isinstance(x, str) else x)
# Remove short words
df_post['cleaned_post4'] = df_post['cleaned_post3'].apply(lambda x: remove_short_words(x) if isinstance(x, str) else x)
# Strip any extra spaces left after removing short words
df_post['cleaned_post4'] = df_post['cleaned_post4'].str.strip().replace(r'\s+', ' ', regex=True)
# Tokenise and remove stopwords
df_post['cleaned_post5'] = df_post['cleaned_post4'].apply(rm_stopwords)
# Measure sentiment score of posts
df_post['senti_score'] = df_post['cleaned_post5'].apply(cal_text_sentiscore)
# Label sentiment of posts
df_post['post_senti'] = df_post['senti_score'].apply(lambda x: 'pos' if x > 0.05 else ('neg' if x < -0.05 else 'neutral'))

# Perform the join operation using 'post' as the key and selecting only 'post' and 'sentiment' from df_post
data = data.merge(df_post[['post', 'senti_score', 'post_senti']], on='post', how='left')


"""
Export the updated DataFrame to excel file
"""
data.to_excel('processed_merged_data2.xlsx', index=False)

df_post.to_excel('processed_posts.xlsx', index=False)



###############################################################################
#----------------------------  Modelling section  ----------------------------#
###############################################################################

"""
Classification Models
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Remove rows of missing value in column 'cmt_final_senti'
data = data.dropna(subset=['cmt_final_senti'])

# Add a feature for the length of the text (in number of characters)
data['cmt_text_len'] = data['cmt_text5'].apply(len)

# Add noise to text sentiment scores for robustness
import numpy as np
np.random.seed(42)
noise_level = 0.3
data['noisy_text_senti'] = data['text_senti_score'] + np.random.normal(0, noise_level, size=len(data))

# Initialize the TF-IDF vectorizer for text
tfidf_vectorizer_optimized = TfidfVectorizer(max_features=500, stop_words='english')

# Fit and transform the text column to create TF-IDF features
tfidf_features_optimized = tfidf_vectorizer_optimized.fit_transform(data['cmt_text5'])

# Combine TF-IDF features, noisy sentiment scores, emoji sentiment, and text length into a single feature matrix
X_combined_sparse = hstack([tfidf_features_optimized, data[['noisy_text_senti', 'emo_senti_score', 'cmt_text_len']]])

# Labels (sentiment class) for classification
y = data['cmt_final_senti']


"""
Modelling for Merged Data
"""
# Split the data into training and testing sets (80% training, 20% testing)
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(X_combined_sparse, y, test_size=0.2, random_state=42)

# Replace NaN values with 0 (since NaN is not allowed in Naive Bayes)
X_train_sparse.data[np.isnan(X_train_sparse.data)] = 0
X_test_sparse.data[np.isnan(X_test_sparse.data)] = 0


# ---- SVM Model ----
# Scaling numeric features (e.g., noisy_polarity_score, text_length)
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_sparse)
X_test_scaled = scaler.transform(X_test_sparse)

svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train_sparse)

# Make predictions and generate classification report for SVM
y_pred_svm = svm_model.predict(X_test_scaled)
svm_report = classification_report(y_test_sparse, y_pred_svm)
print("SVM Classification Report:\n", svm_report)

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test_sparse, y_pred_svm)

# ---- Random Forest Model ----
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_sparse, y_train_sparse)

# Make predictions and generate classification report for Random Forest
y_pred_rf = rf_model.predict(X_test_sparse)
rf_report = classification_report(y_test_sparse, y_pred_rf)
print("Random Forest Classification Report:\n", rf_report)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test_sparse, y_pred_rf)

# ---- Decision Tree Model ----
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model
dt_model.fit(X_train_sparse, y_train_sparse)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test_sparse)

# Generate the classification report for the Decision Tree model
dt_report = classification_report(y_test_sparse, y_pred_dt)
print("Decision Tree Classification Report:\n", dt_report)

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test_sparse, y_pred_dt)


"""
Modelling for Instagram Data
"""
# Get the indices of training and testing data aligned with the "source" column in `data`
train_indices = y_train_sparse.index
test_indices = y_test_sparse.index

# Subset for IG (Instagram) data
ig_train_mask = data.loc[train_indices, 'source'] == 'IG'
ig_test_mask = data.loc[test_indices, 'source'] == 'IG'

# Filter X_train_scaled and X_test_scaled using these masks
X_train_scaled_ig = X_train_scaled[ig_train_mask]
X_test_scaled_ig = X_test_scaled[ig_test_mask]

# Subset the target labels (y_train_sparse and y_test_sparse) for IG
y_train_ig = y_train_sparse[ig_train_mask]
y_test_ig = y_test_sparse[ig_test_mask]

# ------ Train SVM model on IG subset
svm_model_ig = SVC(random_state=42)
svm_model_ig.fit(X_train_scaled_ig, y_train_ig)

# Make predictions and evaluate performance for IG
y_pred_ig = svm_model_ig.predict(X_test_scaled_ig)
svm_report_ig = classification_report(y_test_ig, y_pred_ig)
print("SVM Classification Report for Instagram (IG):\n", svm_report_ig)

# ---- Random Forest Model on IG subset ----
rf_model_ig = RandomForestClassifier(random_state=42)
rf_model_ig.fit(X_train_scaled_ig, y_train_ig)

# Make predictions and generate classification report for Random Forest
y_pred_rf_ig = rf_model_ig.predict(X_test_scaled_ig)
rf_report_ig = classification_report(y_test_ig, y_pred_rf_ig)
print("Random Forest Classification Report for Instagram (IG):\n", rf_report_ig)

# ---- Decision Tree Model on IG subset ----
dt_model_ig = DecisionTreeClassifier(random_state=42)
dt_model_ig.fit(X_train_scaled_ig, y_train_ig)

# Make predictions on the test set
y_pred_dt_ig = dt_model.predict(X_test_scaled_ig)
dt_report_ig = classification_report(y_test_ig, y_pred_dt_ig)
print("Decision Tree Classification Report for Instagram (IG):\n", dt_report_ig)


"""
Modelling for Facebook Data
"""
# Subset for FB (Facebook) data
fb_train_mask = data.loc[train_indices, 'source'] == 'FB'
fb_test_mask = data.loc[test_indices, 'source'] == 'FB'

# Filter X_train_scaled and X_test_scaled using these masks
X_train_scaled_fb = X_train_scaled[fb_train_mask]
X_test_scaled_fb = X_test_scaled[fb_test_mask]

# Subset the target labels (y_train_sparse and y_test_sparse) for IG
y_train_fb = y_train_sparse[fb_train_mask]
y_test_fb = y_test_sparse[fb_test_mask]

# ------ Train SVM model on FB subset
svm_model_fb = SVC(random_state=42)
svm_model_fb.fit(X_train_scaled_fb, y_train_fb)

# Make predictions and evaluate performance for FB
y_pred_svm_fb = svm_model_fb.predict(X_test_scaled_fb)
svm_report_fb = classification_report(y_test_fb, y_pred_svm_fb)
print("SVM Classification Report for Facebook:\n", svm_report_fb)

# ---- Random Forest Model on FB subset ----
rf_model_fb = RandomForestClassifier(random_state=42)
rf_model_fb.fit(X_train_scaled_fb, y_train_fb)

# Make predictions and generate classification report for Random Forest
y_pred_rf_fb = rf_model_fb.predict(X_test_scaled_fb)
rf_report_fb = classification_report(y_test_fb, y_pred_rf_fb)
print("Random Forest Classification Report for Facebook:\n", rf_report_fb)

# ---- Decision Tree Model on FB subset ----
dt_model_fb = DecisionTreeClassifier(random_state=42)
dt_model_fb.fit(X_train_scaled_fb, y_train_fb)

# Make predictions on the test set
y_pred_dt_fb = dt_model.predict(X_test_scaled_fb)
dt_report_fb = classification_report(y_test_fb, y_pred_dt_fb)
print("Decision Tree Classification Report for Facebook:\n", dt_report_fb)


