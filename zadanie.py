import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te
from googletrans import Translator
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('tweets.csv')

def translate_tweets(df):
    translator = Translator(service_urls=['translate.googleapis.com'])
    translated_tweets = []

    for index, row in df.iterrows():
        tweet_content = row['Content']
        lang = row['Lang']
        translated_tweet = tweet_content

        if lang != 'en':
            try:
                if lang == 'in':
                    translated_tweet = translator.translate(tweet_content, dest='en', src='es').text
                elif lang == 'zh':
                    translated_tweet = translator.translate(tweet_content, dest='en', src='zh-cn').text
                elif lang not in ['zxx', 'und', 'qme', 'qht', 'qst', 'qam']:
                    translated_tweet = translator.translate(tweet_content, dest='en', src=lang).text
            except Exception as e:
                print(f"Error translating tweet at index {index}: {e}")
                continue

            print(f"Translated tweet at index {index}")

        translated_tweets.append({'Content': translated_tweet, 'Date': row['Date']})

    df_translated = pd.DataFrame(translated_tweets)
    df_translated.to_csv('translated_tweets.csv', index=False, encoding='utf-8')
    print("Przetłumaczono i zapisano tweety do pliku translated_tweets.csv")

# translate_tweets(df)

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)

    tokens = word_tokenize(tweet)
    tokens = [token for token in tokens if token.isalpha()]

    stop_words = stopwords.words('english')
    stop_words.extend(['premier', 'league', 'premierleague'])
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed = ' '.join(tokens)
    return preprocessed

def preprocess_tweets():
    df_translated = pd.read_csv('translated_tweets.csv', encoding='utf-8')

    df_translated['Content'] = df_translated['Content'].apply(preprocess_tweet)

    df_translated.to_csv('preprocessed_tweets.csv', index=False, encoding='utf-8')
    print('Przetworzono wszystkie tweety i zapisano je do pliku preprocessed_tweets.csv')

# preprocess_tweets()

print("Tweety po preprocessingu wyglądają następująco:")
df = pd.read_csv('preprocessed_tweets.csv')
print(df)

df['Content'] = df['Content'].astype(str).fillna('')
tweets = [tweet.split() for tweet in df['Content']]

def all_words_plots(tweets):
    all_words = [word for tweet in tweets for word in tweet]
    fd = FreqDist(all_words)

    most_common_words = fd.most_common(10)
    words, counts = zip(*most_common_words)
    wordcloud_all = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(fd)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xlabel('Słowa')
    plt.ylabel('Liczba wystąpień')
    plt.title('10 najczęściej występujących słów')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_all, interpolation='bilinear')
    plt.axis('off')
    plt.title('Chmura tagów dla wszystkich słów')
    plt.show()

# all_words_plots(tweets)

def plots_for_sentiments(tweets):
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(' '.join(tweet)) for tweet in tweets]

    positive_tweets = [tweet for tweet, sentiment in zip(tweets, sentiments) if sentiment['pos'] > 0.5 > sentiment['neu'] and sentiment['pos'] > sentiment['neg']]
    negative_tweets = [tweet for tweet, sentiment in zip(tweets, sentiments) if sentiment['neg'] > 0.5 > sentiment['neu'] and sentiment['neg'] > sentiment['pos']]

    positive_words = [word for tweet in positive_tweets for word in tweet]
    positive_words_freq = FreqDist(positive_words)
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(positive_words_freq)

    negative_words = [word for tweet in negative_tweets for word in tweet]
    negative_words_freq = FreqDist(negative_words)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(negative_words_freq)

    plt.figure(figsize=(15, 5))

    plt.subplot(132)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety z opinią pozytywną')

    plt.subplot(133)
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety z opinią negatywną')

    plt.tight_layout()
    plt.show()

# plots_for_sentiments(tweets)

def plots_for_emotions(tweets):
    def make_wordcloud(tweets):
        wordcloud_text = ' '.join(tweets)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
        return wordcloud

    happy_tweets = []
    sad_tweets = []
    angry_tweets = []
    fear_tweets = []
    surprise_tweets = []

    for tweet in tweets:
        full_tweet = ""
        for word in tweet:
            full_tweet += word + " "
        emotions = te.get_emotion(full_tweet)

        main_emotion = max(emotions, key=emotions.get)

        if main_emotion == 'Happy':
            happy_tweets.append(full_tweet)
        elif main_emotion == 'Sad':
            sad_tweets.append(full_tweet)
        elif main_emotion == 'Angry':
            angry_tweets.append(full_tweet)
        elif main_emotion == 'Fear':
            fear_tweets.append(full_tweet)
        elif main_emotion == 'Surprise':
            surprise_tweets.append(full_tweet)

    happy_wc = make_wordcloud(happy_tweets)
    sad_wc = make_wordcloud(sad_tweets)
    angry_wc = make_wordcloud(angry_tweets)
    fear_wc = make_wordcloud(fear_tweets)
    surprise_wc = make_wordcloud(surprise_tweets)

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(happy_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety "Happy"')

    plt.subplot(132)
    plt.imshow(sad_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety "Sad"')

    plt.subplot(133)
    plt.imshow(angry_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety "Angry"')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(fear_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety "Fear"')

    plt.subplot(132)
    plt.imshow(surprise_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Tweety "Surprise"')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))

    happy_words = [word for tweet in happy_tweets for word in tweet.split()]
    happy_words_freq = FreqDist(happy_words)
    axes[0].bar(*zip(*happy_words_freq.most_common(10)))
    axes[0].set_title('Top 10 słów - tweety z opinią happy')
    axes[0].set_xlabel('Słowo')
    axes[0].set_ylabel('Liczba wystąpień')

    sad_words = [word for tweet in sad_tweets for word in tweet.split()]
    sad_words_freq = FreqDist(sad_words)
    axes[1].bar(*zip(*sad_words_freq.most_common(10)))
    axes[1].set_title('Top 10 słów - tweety z opinią sad')
    axes[1].set_xlabel('Słowo')
    axes[1].set_ylabel('Liczba wystąpień')

    angry_words = [word for tweet in angry_tweets for word in tweet.split()]
    angry_words_freq = FreqDist(angry_words)
    axes[2].bar(*zip(*angry_words_freq.most_common(10)))
    axes[2].set_title('Top 10 słów - tweety z opinią angry')
    axes[2].set_xlabel('Słowo')
    axes[2].set_ylabel('Liczba wystąpień')

    fear_words = [word for tweet in fear_tweets for word in tweet.split()]
    fear_words_freq = FreqDist(fear_words)
    axes[3].bar(*zip(*fear_words_freq.most_common(10)))
    axes[3].set_title('Top 10 słów - tweety z opinią fear')
    axes[3].set_xlabel('Słowo')
    axes[3].set_ylabel('Liczba wystąpień')

    surprise_words = [word for tweet in surprise_tweets for word in tweet.split()]
    surprise_words_freq = FreqDist(surprise_words)
    axes[4].bar(*zip(*surprise_words_freq.most_common(10)))
    axes[4].set_title('Top 10 słów - tweety z opinią surprise')
    axes[4].set_xlabel('Słowo')
    axes[4].set_ylabel('Liczba wystąpień')

    plt.tight_layout()
    plt.show()

# plots_for_emotions(tweets)

def wykres_emocji_w_czasie_t2e(df):
    emotions = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
    progress_interval = 1000

    dates_emotions = defaultdict(lambda: defaultdict(list))
    progress_counter = 0

    for index, row in df.iterrows():
        tweet = row['Content']
        date = row['Date']
        
        emotion_scores = te.get_emotion(tweet)
        
        for emotion in emotions:
            dates_emotions[date][emotion].append(emotion_scores[emotion])
        
        progress_counter += 1
        if progress_counter % progress_interval == 0:
            print(f'Przetworzono emocje dla {progress_counter} tweetów')


    dates = list(dates_emotions.keys())

    plt.figure(figsize=(12, 6))

    for emotion in emotions:
        average_emotions = []
        for date in dates:
            scores = dates_emotions[date][emotion]
            if scores:
                average_emotions.append(sum(scores) / len(scores))
            else:
                average_emotions.append(0)
        plt.plot(dates, average_emotions, label=emotion)

    plt.xlabel('Data')
    plt.ylabel('Średnia ocena emocji')
    plt.title('Analiza czasowa emocji')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# wykres_emocji_w_czasie_t2e(df)

def wykres_emocji_w_czasie_vader(df):
    emotions = ['positive', 'negative', 'neutral']
    progress_interval = 1000

    dates_emotions = defaultdict(lambda: defaultdict(list))
    progress_counter = 0

    sia = SentimentIntensityAnalyzer()

    for index, row in df.iterrows():
        tweet = row['Content']
        date = row['Date']

        sentiment_scores = sia.polarity_scores(tweet)

        if sentiment_scores['pos'] > sentiment_scores['neg'] and sentiment_scores['pos'] > sentiment_scores['neu']:
            dates_emotions[date]['positive'].append(sentiment_scores['pos'])

        if sentiment_scores['neg'] > sentiment_scores['pos'] and sentiment_scores['neg'] > sentiment_scores['neu']:
            dates_emotions[date]['negative'].append(sentiment_scores['neg'])

        dates_emotions[date]['neutral'].append(sentiment_scores['neu'])

        progress_counter += 1
        if progress_counter % progress_interval == 0:
            print(f'Przetworzono emocje dla {progress_counter} tweetów')

    dates = list(dates_emotions.keys())

    plt.figure(figsize=(12, 6))

    for emotion in emotions:
        averaged_emotions = []
        for date in dates:
            scores = dates_emotions[date][emotion]
            if scores:
                averaged_emotions.append(sum(scores) / len(scores))
            else:
                averaged_emotions.append(0)
        plt.plot(dates, averaged_emotions, label=emotion)

    plt.xlabel('Data')
    plt.ylabel('Średnia ocena emocji')
    plt.title('Analiza czasowa emocji')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# wykres_emocji_w_czasie_vader(df)

def analiza_tematyki_i_klasteryzacja(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Content'])

    # Analiza tematyki (LDA)
    lda = LatentDirichletAllocation(n_components=6, random_state=0)
    lda.fit(X)

    # Klasteryzacja (K-Means)
    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(X)

    # Dodanie klastrów do DataFrame
    df['Cluster'] = clusters

    # Wizualizacja klastrów
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=df)
    plt.title('Liczba dokumentów w klastrach')
    plt.show()

    # Funkcja do wyświetlania chmur słów
    def plot_word_clouds(model, feature_names, no_top_words):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            wordcloud = WordCloud(background_color='white',
                                width=800, 
                                height=400,
                                max_words=no_top_words).generate_from_frequencies({feature_names[i]: topic[i] for i in topic.argsort()[:-no_top_words - 1:-1]})
            ax = axes[topic_idx]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Topic {topic_idx+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    plot_word_clouds(lda, vectorizer.get_feature_names_out(), 10)

    # Funkcja do wyświetlania topowych słów na wykresach słupkowych
    def plot_top_words(model, feature_names, n_top_words, title):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Topic {topic_idx+1}', fontdict={'fontsize': 15})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=12)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=20)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

    plot_top_words(lda, vectorizer.get_feature_names_out(), 10, 'Top words per topic')

    # Wizualizacja tematów w klastrach
    df['Topic'] = lda.transform(X).argmax(axis=1)
    plt.figure(figsize=(14, 8))
    sns.countplot(x='Cluster', hue='Topic', data=df)
    plt.title('Distribution of Topics in Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Topic', loc='upper right')
    plt.show()

# analiza_tematyki_i_klasteryzacja(df)