# import json
# from flask import Flask, jsonify, request
# from collections import defaultdict
# from datetime import datetime, timedelta
# import requests
# from flask_cors import CORS,cross_origin
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from concurrent.futures import ThreadPoolExecutor
# import redis
# from keybert import KeyBERT

# app = Flask(__name__)
# CORS(app)


# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# # API endpoint to accept the body in the specified format
# api_url = "https://api.apify.com/v2/acts/junglee~amazon-reviews-scraper/run-sync-get-dataset-items?token=apify_api_PIVg196KLR7NEJSweVbBkYVp46Z0vV23w0vP"



# @app.route('/scrape', methods=['POST'])
# @cross_origin()
# def scrape():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     # Check if formatted data exists in Redis cache
#     cached_data = redis_client.get(url)
#     if cached_data:
#         return cached_data.decode('utf-8')

#     # Data not found in cache, scrape and process it
#     scraped_data = scrape_and_process_data(url)

#     # Convert scraped data to JSON string
#     scraped_data_json = json.dumps(scraped_data)

#     # Cache the formatted data in Redis with expiration time of 7 days
#     redis_client.setex(url, timedelta(days=7), scraped_data_json)

#     return jsonify(scraped_data)

# def scrape_and_process_data(url):
#     # Your scraping logic here
#     response = requests.post(api_url, json={"productUrls": [{"url": url}]})
#     scraped_data = response.json()

#     # Process and format the scraped data
#     formatted_data = process_data(scraped_data)

#     return formatted_data

# def process_data(reviews):
#     # Initialize variables for calculations
#     total_reviews = 0
#     total_ratings = 0
#     csat_count = 0
#     verified_count = 0
#     rating_counts = defaultdict(int)
#     rating_trend = defaultdict(list)

#     # Iterate through each review
#     for review in reviews:
#         # Increment total reviews count
#         total_reviews += 1
#         # Increment total ratings count
#         total_ratings += review["ratingScore"]
#         # Increment CSAT count if rating is 4 or 5
#         if review["ratingScore"] >= 4:
#             csat_count += 1
#         # Increment verified count if review is verified
#         if review["isVerified"]:
#             verified_count += 1
#         # Increment rating count for the corresponding rating score
#         rating_counts[review["ratingScore"]] += 1
#         # Append rating score to rating trend for the corresponding date
#         rating_trend[review["date"]].append(review["ratingScore"])

#     # Calculate average rating
#     avg_rating = total_ratings / total_reviews if total_reviews > 0 else 0

#     # Calculate CSAT (Customer Satisfaction) rate in percentage
#     csat_rate = (csat_count / total_reviews) * 100 if total_reviews > 0 else 0

#     # Sort rating counts dictionary
#     # max_rating = max(rating_counts.keys(), default=0)
#     # for i in range(1, max_rating + 1):
#     #     if i not in rating_counts:
#     #         rating_counts[i] = 0

#     # # Sort rating counts dictionary
#     # sorted_rating_counts = dict(sorted(rating_counts.items()))
#     max_rating = max(rating_counts.keys(), default=0)
#     formatted_rating_counts = [{rating: rating_counts.get(rating, 0)} for rating in range(1, max_rating + 1)]

#     # Format rating trend by sorting by date
#     formatted_rating_trend = sorted([(date, round(sum(ratings) / len(ratings), 2)) for date, ratings in rating_trend.items()], key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

#     # Return the processed data
#     return {
#         "total_reviews": total_reviews,
#         "avg_rating": round(avg_rating, 2),
#         "csat_rate": round(csat_rate, 2),
#         "total_verified_count": verified_count,
#         "rating_counts": formatted_rating_counts,
#         "rating_trend": formatted_rating_trend
#     }
    
# @app.route('/',methods=['GET'])
# @cross_origin()
# def home():
#     return "Welcome to Amazon Review Scraper API"



# # Load the three different pipelines
# sentiment_pipeline = pipeline("text-classification", model='nlptown/bert-base-multilingual-uncased-sentiment', tokenizer='nlptown/bert-base-multilingual-uncased-sentiment')
# emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', tokenizer='bhadresh-savani/bert-base-uncased-emotion')
# zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
# sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

# executor = ThreadPoolExecutor(max_workers=10)  # Adjust the number of workers as needed

# @app.route('/classify', methods=['POST'])
# @cross_origin()
# def classify_text():
#     try:
#         data = request.get_json()
#         texts = data.get('texts', [])

#         if not texts:
#             return jsonify({'error': 'Texts field is required and must be a non-empty array'}), 400

#         results = []

#         with executor as pool:
#             futures = {pool.submit(process_text, text): text for text in texts}
#             for future in futures:
#                 text = futures[future]
#                 try:
#                     result = future.result()
#                     results.append(result)
#                 except Exception as e:
#                     results.append({'text': text, 'error': str(e)})

#         return jsonify(results), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# def process_text(text):
#     if not text:
#         return {'text': text, 'error': 'Empty text provided'}
    
#     # Perform sentiment analysis
#     sentiment_prediction = sentiment_pipeline(text)

#     # Perform emotion analysis
#     emotion_prediction = emotion_pipeline(text)

#     # Perform zero-shot classification
#     zero_shot_output = zero_shot_classifier(text, ["effectiveness", "ease of use", "comfort", "value for money"], multi_label=False)

#     return {
#         'text': text,
#         'sentiment_prediction': sentiment_prediction,
#         'emotion_prediction': emotion_prediction,
#         'zero_shot_output': zero_shot_output
#     }
    
    
# kw_model = KeyBERT(model='all-MiniLM-L6-v2')
# @app.route('/extract_keywords', methods=['POST'])
# @cross_origin()
# def keyWords():
#     try:
#         # Get the input texts from the request
#         data = request.get_json()
#         texts = data.get('texts', [])

#         # Combine all reviews into a single text
#         combined_text = ' '.join(texts)

#         # Extract keywords from the combined text
#         print('start model')
#         keywords = kw_model.extract_keywords(combined_text)
#         print('end model')
#         print('keywords are:',keywords)
#         # Initialize arrays for positive and negative keywords
#         positive_keywords = []
#         negative_keywords = []

#         # Classify sentiment for each keyword and store in respective arrays
#         for keyword in keywords:
#             result = sentiment_analysis(keyword)
#             sentiment = result[0]['label']
#             print(sentiment)
            
#             if sentiment == 'POSITIVE':
#                 positive_keywords.append(keyword)
#             elif sentiment == 'NEGATIVE':
#                 negative_keywords.append(keyword)

#         # Return the classified sentiment results as JSON
#         return jsonify({
#             "positive_keywords": positive_keywords,
#             "negative_keywords": negative_keywords})

#     except Exception as e:
#         return jsonify({"error":str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)











# import json
# from flask import Flask, jsonify, request
# from collections import defaultdict
# from datetime import datetime, timedelta
# import requests
# from flask_cors import CORS, cross_origin
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from concurrent.futures import ThreadPoolExecutor
# import redis
# from keybert import KeyBERT

# app = Flask(__name__)
# CORS(app)

# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
# api_url = "https://api.apify.com/v2/acts/junglee~amazon-reviews-scraper/run-sync-get-dataset-items?token=apify_api_PIVg196KLR7NEJSweVbBkYVp46Z0vV23w0vP"

# # Load the three different pipelines
# sentiment_pipeline = pipeline("text-classification", model='nlptown/bert-base-multilingual-uncased-sentiment', tokenizer='nlptown/bert-base-multilingual-uncased-sentiment')
# emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', tokenizer='bhadresh-savani/bert-base-uncased-emotion')
# zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
# sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")

# executor = ThreadPoolExecutor(max_workers=10)



# @app.route('/scrape', methods=['POST'])
# @cross_origin()
# def scrape():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     # Check if data exists in Redis cache
#     cached_data = redis_client.get(url)
#     if cached_data:
#         return cached_data.decode('utf-8')
#     # if redis_client.exists(url):
#     #     # Delete the cached data for the URL
#     #     redis_client.delete(url)

#     # Data not found in cache, scrape and process it
#     response = requests.post(api_url, json={"productUrls": [{"url": url}]})
#     scraped_data = response.json()

#     # Process and format the scraped data
#     formatted_data = process_data(scraped_data)

#     # Cache the formatted data in Redis with expiration time of 7 days
#     redis_client.setex(url, timedelta(days=7), json.dumps(formatted_data))

#     return jsonify(formatted_data)

# def process_data(reviews):
#     total_reviews = 0
#     total_ratings = 0
#     csat_count = 0
#     verified_count = 0
#     rating_counts = defaultdict(int)
#     rating_trend = defaultdict(list)
#     review_descriptions = []

#     for review in reviews:
#         total_reviews += 1
#         total_ratings += review["ratingScore"]
#         if review["ratingScore"] >= 4:
#             csat_count += 1
#         if review["isVerified"]:
#             verified_count += 1
#         rating_counts[review["ratingScore"]] += 1
#         rating_trend[review["date"]].append(review["ratingScore"])
#         review_descriptions.append(review["reviewDescription"])

#     avg_rating = total_ratings / total_reviews if total_reviews > 0 else 0
#     csat_rate = (csat_count / total_reviews) * 100 if total_reviews > 0 else 0

#     max_rating = max(rating_counts.keys(), default=0)
#     formatted_rating_counts = [{rating: rating_counts.get(rating, 0)} for rating in range(1, max_rating + 1)]
#     formatted_rating_trend = sorted([(date, round(sum(ratings) / len(ratings), 2)) for date, ratings in rating_trend.items()], key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

#     # Classify the review descriptions
#     classified_reviews = classify_text(review_descriptions[:50])

#     return {
#         "total_reviews": total_reviews,
#         "avg_rating": round(avg_rating, 2),
#         "csat_rate": round(csat_rate, 2),
#         "total_verified_count": verified_count,
#         "rating_counts": formatted_rating_counts,
#         "rating_trend": formatted_rating_trend,
#         "classified_reviews": classified_reviews
#     }

# def classify_text(texts):
#     results = []

#     with executor as pool:
#         futures = {pool.submit(process_text, text): text for text in texts}
#         for future in futures:
#             text = futures[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 results.append({'text': text, 'error': str(e)})

#     return results

# def process_text(text):
#     if not text:
#         return {'text': text, 'error': 'Empty text provided'}
    
#     sentiment_prediction = sentiment_pipeline(text)
#     emotion_prediction = emotion_pipeline(text)
#     zero_shot_output = zero_shot_classifier(text, ["effectiveness", "ease of use", "comfort", "value for money"], multi_label=False)

#     return {
#         'text': text,
#         'sentiment_prediction': sentiment_prediction,
#         'emotion_prediction': emotion_prediction,
#         'zero_shot_output': zero_shot_output
#     }

# if __name__ == '__main__':
#     app.run(debug=True)
















import json
from flask import Flask, jsonify, request
from collections import defaultdict
from datetime import datetime, timedelta
import requests
from flask_cors import CORS, cross_origin
from transformers import pipeline
import redis
from keybert import KeyBERT
# from youtube_transcript_api import YouTubeTranscriptApi


app = Flask(__name__)
CORS(app)

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
api_url = "https://api.apify.com/v2/acts/junglee~amazon-reviews-scraper/run-sync-get-dataset-items?token=apify_api_PIVg196KLR7NEJSweVbBkYVp46Z0vV23w0vP"

sentiment_pipeline = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
emotion_pipeline = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', tokenizer='bhadresh-savani/bert-base-uncased-emotion')
zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def scrape_and_process_data(url):
    cached_data = redis_client.get(url)
    # if cached_data:
    #     return cached_data.decode('utf-8')
    if cached_data:
        # formatted_data = cached_data.decode('utf-8')
        # Retrieve reviews from Redis
        reviews = redis_client.get(f"{url}_reviews")
        if reviews:
            print("Reviews from Redis:", reviews.decode('utf-8'))
        # return jsonify(json.loads(cached_data.decode('utf-8')))
        return json.loads(cached_data.decode('utf-8'))
    # results = redis_client.get(f"{url}_sentiment")
    # if results:
    #     return jsonify(json.loads(results.decode('utf-8')))
    # if redis_client.exists(url):
    # #     # Delete the cached data for the URL
    #     redis_client.delete(url)

    response = requests.post(api_url, json={"productUrls": [{"url": url}]})
    scraped_data = response.json()
    formatted_data, review_descriptions = process_data(scraped_data)
    
    redis_client.setex(url, timedelta(days=7), json.dumps(formatted_data))
    redis_client.setex(f"{url}_reviews", timedelta(days=7), json.dumps(review_descriptions[:30]))

    return formatted_data

def process_data(reviews):
    total_reviews = 0
    total_ratings = 0
    csat_count = 0
    verified_count = 0
    rating_counts = defaultdict(int)
    rating_trend = defaultdict(list)
    review_descriptions = []

    for review in reviews:
        total_reviews += 1
        total_ratings += review["ratingScore"]
        if review["ratingScore"] >= 4:
            csat_count += 1
        if review["isVerified"]:
            verified_count += 1
        rating_counts[review["ratingScore"]] += 1
        rating_trend[review["date"]].append(review["ratingScore"])
        review_descriptions.append(review["reviewDescription"])

    avg_rating = total_ratings / total_reviews if total_reviews > 0 else 0
    csat_rate = (csat_count / total_reviews) * 100 if total_reviews > 0 else 0

    max_rating = max(rating_counts.keys(), default=0)
    formatted_rating_counts = [{rating: rating_counts.get(rating, 0)} for rating in range(1, max_rating + 1)]
    formatted_rating_trend = sorted([(date, round(sum(ratings) / len(ratings), 2)) for date, ratings in rating_trend.items()], key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

    return {
        "total_reviews": total_reviews,
        "avg_rating": round(avg_rating, 2),
        "csat_rate": round(csat_rate, 2),
        "total_verified_count": verified_count,
        "rating_counts": formatted_rating_counts,
        "rating_trend": formatted_rating_trend
    }, review_descriptions

@app.route('/scraping', methods=['POST'])
@cross_origin()
def scraping():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL not provided"})

    formatted_data = scrape_and_process_data(url)

    return jsonify(formatted_data)

# @app.route('/sentiment-analysis', methods=['POST'])
# @cross_origin()
# def sentiment_analysis():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     reviews = redis_client.get(f"{url}_reviews")
#     if not reviews:
#         return jsonify({"error": "Reviews not found for the URL"})

#     reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
#     results = sentiment_pipeline(reviews_list)  # Pass the reviews to the sentiment analysis pipeline
#     return jsonify(results)

# @app.route('/emotion-analysis', methods=['POST'])
# @cross_origin()
# def emotion_analysis():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     reviews = redis_client.get(f"{url}_reviews")
#     if not reviews:
#         return jsonify({"error": "Reviews not found for the URL"})

#     reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
#     results = emotion_pipeline(reviews_list)  # Pass the reviews to the emotion analysis pipeline
#     return jsonify(results)

# @app.route('/zero-shot-analysis', methods=['POST'])
# @cross_origin()
# def zero_shot_analysis():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     reviews = redis_client.get(f"{url}_reviews")
#     if not reviews:
#         return jsonify({"error": "Reviews not found for the URL"})

#     reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
#     results = zero_shot_classifier(reviews_list, ["effectiveness", "ease of use", "comfort", "value for money"], multi_label=False)  # Pass the reviews to the zero-shot analysis pipeline
#     return jsonify(results)

@app.route('/sentiment-analysis', methods=['POST'])
@cross_origin()
def sentiment_analysis():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL not provided"})

    # Check if sentiment analysis results exist in Redis for the given URL
    results = redis_client.get(f"{url}_sentiment")
    if results:
        return jsonify(json.loads(results.decode('utf-8')))

    # Retrieve reviews from Redis
    reviews = redis_client.get(f"{url}_reviews")
    if not reviews:
        return jsonify({"error": "Reviews not found for the URL"})

    reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
    sentiment_results = sentiment_pipeline(reviews_list)  # Run sentiment analysis
    redis_client.setex(f"{url}_sentiment", timedelta(days=7), json.dumps(sentiment_results))  # Save results in Redis

    return jsonify(sentiment_results)


@app.route('/emotion-analysis', methods=['POST'])
@cross_origin()
def emotion_analysis():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL not provided"})

    # Check if emotion analysis results exist in Redis for the given URL
    results = redis_client.get(f"{url}_emotion")
    if results:
        return jsonify(json.loads(results.decode('utf-8')))

    # Retrieve reviews from Redis
    reviews = redis_client.get(f"{url}_reviews")
    if not reviews:
        return jsonify({"error": "Reviews not found for the URL"})

    reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
    emotion_results = emotion_pipeline(reviews_list)  # Run emotion analysis
    redis_client.setex(f"{url}_emotion", timedelta(days=7), json.dumps(emotion_results))  # Save results in Redis

    return jsonify(emotion_results)


@app.route('/zero-shot-analysis', methods=['POST'])
@cross_origin()
def zero_shot_analysis():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL not provided"})

    # Check if zero-shot analysis results exist in Redis for the given URL
    results = redis_client.get(f"{url}_zero_shot")
    if results:
        return jsonify(json.loads(results.decode('utf-8')))

    # Retrieve reviews from Redis
    reviews = redis_client.get(f"{url}_reviews")
    if not reviews:
        return jsonify({"error": "Reviews not found for the URL"})

    reviews_list = json.loads(reviews.decode('utf-8'))  # Deserialize reviews from Redis
    zero_shot_results = zero_shot_classifier(reviews_list, ["effectiveness", "ease of use", "comfort", "value for money"], multi_label=False)  # Run zero-shot analysis
    redis_client.setex(f"{url}_zero_shot", timedelta(days=7), json.dumps(zero_shot_results))  # Save results in Redis

    return jsonify(zero_shot_results)



kw_model = KeyBERT(model='all-MiniLM-L6-v2')
# @app.route('/extract_keywords', methods=['POST'])
# @cross_origin()
# def keyWords():
#         # Get the input texts from the request
#         data = request.get_json()
#         texts = data.get('texts', [])

#         # Combine all reviews into a single text
#         combined_text = ' '.join(texts)

#         # Extract keywords from the combined text
#         print('start model')
#         keywords = kw_model.extract_keywords(combined_text)
#         print('end model')
#         print('keywords are:',keywords)
#         return jsonify({ 'keywords': keywords})

@app.route('/extract_keywords', methods=['POST'])
@cross_origin()
def extract_keywords():
    # Get the input URL from the request
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL not provided"})

    # Check if keywords are already cached in Redis for this URL
    cached_keywords = redis_client.get(f"{url}_keywords")
    if cached_keywords:
        keywords = json.loads(cached_keywords.decode('utf-8'))
        return jsonify({'keywords': keywords})

    # Fetch reviews from Redis for the given URL
    cached_reviews = redis_client.get(f"{url}_reviews")
    if not cached_reviews:
        return jsonify({"error": "No reviews found for the provided URL"})

    # Combine all reviews into a single text
    reviews = json.loads(cached_reviews.decode('utf-8'))
    combined_text = ' '.join(reviews)

    # Extract keywords from the combined text using KeyBERT
    keywords = kw_model.extract_keywords(combined_text)

    # Save extracted keywords to Redis with expiration time of 7 days
    redis_client.setex(f"{url}_keywords", timedelta(days=7), json.dumps(keywords))

    return jsonify({'keywords': keywords})

# @app.route('/transcribe', methods=['GET'])
# @cross_origin()
# async def  transcribe():
    
    
    
#     result = await YouTubeTranscriptApi.get_transcript('1sRaLqtHXQU')
    
#     return jsonify({'transcript': result})


    

if __name__ == '__main__':
    app.run(debug=True)






# @app.route('/scrape', methods=['POST'])
# @cross_origin()
# def scrape():
#     data = request.json
#     url = data.get("url")
#     if not url:
#         return jsonify({"error": "URL not provided"})

#     # Scrape data from the URL
#     response = requests.post(api_url, json={"productUrls": [{"url": url}]})
#     scraped_data = response.json()

#     # Format the scraped data
#     formatted_data = process_data(scraped_data)

#     return jsonify(formatted_data)




# @app.route('/classify', methods=['POST'])
# @cross_origin()
# def classify_text():
#     try:
#         data = request.get_json()
#         texts = data.get('texts', [])

#         if not texts:
#             return jsonify({'error': 'Texts field is required and must be a non-empty array'}), 400

#         results = []

#         for text in texts:
#             if not text:
#                 results.append({'error': 'Empty text provided'})
#                 continue

#             # Perform sentiment analysis
#             sentiment_prediction = sentiment_pipeline(text)

#             # Perform emotion analysis
#             emotion_prediction = emotion_pipeline(text)

#             # Perform zero-shot classification
#             zero_shot_output = zero_shot_classifier(text, ["politics", "economy", "entertainment", "environment"], multi_label=False)

#             results.append({
#                 'text': text,
#                 'sentiment_prediction': sentiment_prediction,
#                 'emotion_prediction': emotion_prediction,
#                 'zero_shot_output': zero_shot_output
#             })

#         return jsonify(results), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

