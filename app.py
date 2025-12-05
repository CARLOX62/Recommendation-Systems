from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import os
from werkzeug.utils import secure_filename
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Disable oneDNN optimizations to avoid potential issues with TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load models and data
# Books
popular_df = pickle.load(open('Book Recommendation System/popular_books.pkl', 'rb'))
pt = pickle.load(open('Book Recommendation System/pt.pkl', 'rb'))
books = pickle.load(open('Book Recommendation System/books.pkl', 'rb'))
similarity_scores = pickle.load(open('Book Recommendation System/similarity_scores.pkl', 'rb'))

# Movies
movies = pickle.load(open('Movie Recommendation System/movies.pkl', 'rb'))
# Compute similarity for movies if not saved
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
movie_vectors = tfidf.fit_transform(movies['tags']).toarray()
movie_similarity = cosine_similarity(movie_vectors)

def fetch_poster(movie_id):
    try:
        url = "https://api.themoviedb.org/3/movie/{}?api_key=7bf081a707237b410cefe888c82f7b85&language=en-US".format(movie_id)
        data = requests.get(url, timeout=30)
        data.raise_for_status()  # Raise an exception for bad status codes
        data = data.json()
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = "https://image.tmdb.org/t/p/w300/" + poster_path
        else:
            full_path = "https://via.placeholder.com/300x450?text=No+Image"
        rating = data.get('vote_average', 'N/A')
        if rating == 'N/A' or not isinstance(rating, (int, float)):
            rating = 'N/A'
        else:
            rating = round(rating, 1)
        return full_path, rating
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching poster for movie_id {movie_id}")
        return "https://via.placeholder.com/300x450?text=Timeout", 'N/A'
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching poster for movie_id {movie_id}: {e}")
        return "https://via.placeholder.com/300x450?text=Error", 'N/A'
    except Exception as e:
        print(f"Unexpected error fetching poster for movie_id {movie_id}: {e}")
        return "https://via.placeholder.com/300x450?text=Error", 'N/A'

# Music
music_df = pickle.load(open('Music Recommendation System/df.pkl', 'rb'))
music_similarity = pickle.load(open('Music Recommendation System/similarity.pkl', 'rb'))

CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

# Fashion
fashion_embeddings = np.array(pickle.load(open('Fashion Recommendation System/embeddings.pkl', 'rb')))
fashion_filenames = pickle.load(open('Fashion Recommendation System/filenames.pkl', 'rb'))
fashion_neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
fashion_neighbors.fit(fashion_embeddings)

# Fashion model
fashion_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
fashion_model.trainable = False
fashion_model = tf.keras.Sequential([fashion_model, tf.keras.layers.GlobalMaxPooling2D()])

app.config['UPLOAD_FOLDER'] = 'static/uploads'

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

@app.route('/')
def index():
    return render_template('index.html', popular_books=popular_df.head(10).to_dict(orient='records'))

@app.route('/books')
def books_page():
    return render_template('books.html', popular_books=popular_df.head(10).to_dict(orient='records'))

@app.route('/movies')
def movies_page():
    popular_movies = movies.head(10).to_dict(orient='records')
    for movie in popular_movies:
        poster, rating = fetch_poster(movie['movie_id'])
        movie['poster'] = poster
        movie['rating'] = rating
    return render_template('movies.html', popular_movies=popular_movies)

@app.route('/music')
def music_page():
    popular_music = music_df.head(10).to_dict(orient='records')
    for song in popular_music:
        song['poster'] = get_song_album_cover_url(song['song'], song['artist'])
    return render_template('music.html', popular_music=popular_music)

@app.route('/fashion')
def fashion_page():
    popular_fashion = [{'image': url_for('fashion_images', filename=os.path.basename(fashion_filenames[i]))} for i in range(min(10, len(fashion_filenames)))]
    return render_template('fashion.html', popular_fashion=popular_fashion)

@app.route('/fashion_images/<path:filename>')
def fashion_images(filename):
    return send_from_directory('Fashion Recommendation System/images', filename)

@app.route('/recommend_books', methods=['POST'])
def recommend_books():
    book_name = request.form['book_name']
    if book_name not in pt.index:
        return render_template('results.html', category='Books', message='Book not found in database.', recommendations=[])
    
    index = pt.index.tolist().index(book_name)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:11]
    
    data = []
    for i in similar_items:
        item = {}
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item['title'] = list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values)[0]
        item['author'] = list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values)[0]
        item['image'] = list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values)[0]
        data.append(item)
    
    return render_template('results.html', category='Books', recommendations=data)

@app.route('/recommend_movies', methods=['POST'])
def recommend_movies():
    movie_name = request.form['movie_name']
    if movie_name not in movies['title'].values:
        return render_template('results.html', category='Movies', message='Movie not found in database.', recommendations=[])

    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = movie_similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    data = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        poster, rating = fetch_poster(movie_id)
        data.append({'title': movies.iloc[i[0]].title, 'poster': poster, 'rating': rating})

    return render_template('results.html', category='Movies', recommendations=data)

@app.route('/recommend_music', methods=['POST'])
def recommend_music():
    song_name = request.form['song_name']
    if song_name not in music_df['song'].values:
        return render_template('results.html', category='Music', message='Song not found in database.', recommendations=[])
    
    idx = music_df[music_df['song'] == song_name].index[0]
    distances = sorted(list(enumerate(music_similarity[idx])), reverse=True, key=lambda x: x[1])
    
    data = []
    for m_id in distances[1:11]:
        artist = music_df.iloc[m_id[0]].artist
        data.append({'title': music_df.iloc[m_id[0]].song, 'poster': get_song_album_cover_url(music_df.iloc[m_id[0]].song, artist)})
    
    return render_template('results.html', category='Music', recommendations=data)

@app.route('/recommend_fashion', methods=['POST'])
def recommend_fashion():
    if 'image' not in request.files:
        return render_template('results.html', category='Fashion', message='No image uploaded.', recommendations=[])
    
    file = request.files['image']
    if file.filename == '':
        return render_template('results.html', category='Fashion', message='No image selected.', recommendations=[])
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    features = extract_features(filepath, fashion_model)
    distances, indices = fashion_neighbors.kneighbors([features])
    
    data = []
    for file in indices[0][1:6]:
        data.append({'image': url_for('fashion_images', filename=os.path.basename(fashion_filenames[file]))})
    
    return render_template('results.html', category='Fashion', recommendations=data)

if __name__ == '__main__':
    app.run(debug=True)
