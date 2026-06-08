#!/usr/bin/env python3
# Social Media Narrative Analysis - Sample Script
# Demonstrates computational social science skills for digital media research.
# Author: Sidra Jabeen Khan
# Date: June 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# Sample dataset simulating social media posts from political events
sample_posts = [
    {"platform": "Twitter", "text": "The EU must expand to include all European nations for security and prosperity", "event": "EU_enlargement", "engagement": 245},
    {"platform": "Twitter", "text": "EU enlargement is a threat to our national sovereignty and jobs", "event": "EU_enlargement", "engagement": 189},
    {"platform": "Facebook", "text": "Peace dialogue is the only solution to de-escalate tensions in the region", "event": "conflict_resolution", "engagement": 567},
    {"platform": "Facebook", "text": "Military action is necessary when diplomacy fails completely", "event": "conflict_resolution", "engagement": 423},
    {"platform": "Telegram", "text": "Disinformation campaigns are spreading false narratives about the election", "event": "election_integrity", "engagement": 1200},
    {"platform": "Telegram", "text": "The election was rigged and the results cannot be trusted at all", "event": "election_integrity", "engagement": 3400},
    {"platform": "TikTok", "text": "Youth voices matter in shaping European democracy and future", "event": "youth_democracy", "engagement": 8900},
    {"platform": "TikTok", "text": "Young people are being manipulated by political elites for their agenda", "event": "youth_democracy", "engagement": 5600},
    {"platform": "Twitter", "text": "Social cohesion requires trust in institutions and transparent governance", "event": "social_cohesion", "engagement": 334},
    {"platform": "Twitter", "text": "Institutions are corrupt and cannot be trusted by ordinary citizens", "event": "social_cohesion", "engagement": 892},
    {"platform": "Facebook", "text": "Dialogue between opposing groups can bridge divides and build understanding", "event": "polarization", "engagement": 445},
    {"platform": "Facebook", "text": "There is no compromise with those who want to destroy our way of life", "event": "polarization", "engagement": 1200},
    {"platform": "Telegram", "text": "Extremist narratives are being amplified by algorithmic recommendation systems", "event": "extremism", "engagement": 780},
    {"platform": "Telegram", "text": "Radical solutions are needed when the system refuses to change", "event": "extremism", "engagement": 2100},
    {"platform": "TikTok", "text": "Democratic participation empowers citizens to shape their collective future", "event": "democratic_participation", "engagement": 6700},
    {"platform": "TikTok", "text": "Voting changes nothing when the system is controlled by elites", "event": "democratic_participation", "engagement": 4500},
]

df = pd.DataFrame(sample_posts)

# Text preprocessing function
def preprocess_text(text):
    # Clean and normalize social media text
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

# Basic sentiment lexicon (simplified for demonstration)
positive_words = ['peace', 'dialogue', 'prosperity', 'security', 'trust', 'transparent', 
                  'empower', 'future', 'democracy', 'understanding', 'bridge', 'cohesion',
                  'participation', 'matter', 'solution', 'include', 'necessary']
negative_words = ['threat', 'fails', 'rigged', 'corrupt', 'destroy', 'manipulated', 
                  'extremist', 'radical', 'refuses', 'false', 'cannot', 'tensions',
                  'conflict', 'polarization', 'disinformation', 'elites', 'compromise']

def basic_sentiment(text):
    # Calculate basic sentiment score using word counts
    words = text.split()
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)

    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['clean_text'].apply(basic_sentiment)

# Narrative classification based on keywords
def classify_narrative(text):
    # Classify narrative type based on keyword patterns
    text = text.lower()

    if any(word in text for word in ['eu', 'enlargement', 'sovereignty', 'nations']):
        return 'institutional_legitimacy'
    elif any(word in text for word in ['conflict', 'military', 'peace', 'diplomacy', 'tensions']):
        return 'conflict_narrative'
    elif any(word in text for word in ['election', 'rigged', 'disinformation', 'trust']):
        return 'electoral_integrity'
    elif any(word in text for word in ['youth', 'young', 'democracy', 'future']):
        return 'generational_politics'
    elif any(word in text for word in ['cohesion', 'divide', 'bridge', 'understanding', 'polarization']):
        return 'social_polarization'
    elif any(word in text for word in ['extremist', 'radical', 'system', 'change']):
        return 'radicalization_discourse'
    else:
        return 'other'

df['narrative_class'] = df['clean_text'].apply(classify_narrative)

print("=== NARRATIVE CLASSIFICATION RESULTS ===")
print(df[['platform', 'event', 'sentiment', 'narrative_class', 'engagement']].head(10))
print()

# Summary statistics
print("=== SENTIMENT DISTRIBUTION BY PLATFORM ===")
sentiment_platform = pd.crosstab(df['platform'], df['sentiment'])
print(sentiment_platform)
print()

print("=== NARRATIVE CLASS DISTRIBUTION ===")
narrative_counts = df['narrative_class'].value_counts()
print(narrative_counts)
print()

print("=== ENGAGEMENT BY SENTIMENT ===")
engagement_sentiment = df.groupby('sentiment')['engagement'].agg(['mean', 'median', 'std'])
print(engagement_sentiment)
print()

# Word frequency analysis
all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words)
print("=== TOP 15 MOST FREQUENT WORDS ===")
for word, count in word_freq.most_common(15):
    if len(word) > 3:
        print(f"  {word}: {count}")
print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sentiment distribution by platform
sentiment_platform.plot(kind='bar', ax=axes[0,0], color=['green', 'gray', 'red'])
axes[0,0].set_title('Sentiment Distribution by Platform')
axes[0,0].set_xlabel('Platform')
axes[0,0].set_ylabel('Number of Posts')
axes[0,0].legend(title='Sentiment')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Narrative class distribution
narrative_counts.plot(kind='barh', ax=axes[0,1], color='steelblue')
axes[0,1].set_title('Narrative Classification Distribution')
axes[0,1].set_xlabel('Number of Posts')

# 3. Engagement by sentiment
df.boxplot(column='engagement', by='sentiment', ax=axes[1,0])
axes[1,0].set_title('Engagement Levels by Sentiment')
axes[1,0].set_xlabel('Sentiment')
axes[1,0].set_ylabel('Engagement (likes/shares)')

# 4. Top words wordcloud-style bar chart
top_words = [word for word, count in word_freq.most_common(15) if len(word) > 3][:10]
top_counts = [count for word, count in word_freq.most_common(15) if len(word) > 3][:10]
axes[1,1].barh(top_words, top_counts, color='coral')
axes[1,1].set_title('Top 10 Frequent Words (filtered)')
axes[1,1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('narrative_analysis.png', dpi=150)
plt.show()

# Early warning indicator simulation
def calculate_polarization_index(df_subset):
    # Calculate a simple polarization index for a subset of posts
    # Higher values indicate more polarized discourse
    sentiment_counts = df_subset['sentiment'].value_counts(normalize=True)
    neutral_prop = sentiment_counts.get('neutral', 0)
    polarization = 1 - neutral_prop
    avg_engagement = df_subset['engagement'].mean()
    return polarization * (avg_engagement / 1000)

print("=== POLARIZATION INDEX BY EVENT ===")
for event in df['event'].unique():
    subset = df[df['event'] == event]
    pol_index = calculate_polarization_index(subset)
    print(f"  {event}: {pol_index:.3f}")
print()

print("Analysis complete. Visualization saved as 'narrative_analysis.png'")
print("\nNote: This is a demonstration script using synthetic data.")
print("In actual research, data would be collected via platform APIs")
print("and analysis would use more sophisticated NLP techniques (NLTK, spaCy, transformers).")
