import pickle
import nltk
import pandas as pd
import requests
import streamlit as st

nltk.download('punkt')
# summarizing the text
from sumy.summarizers.lex_rank import LexRankSummarizer

# uses the lex rank summarizer
summarizer_lex = LexRankSummarizer()


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    recommended_movies_details = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        # fetch poster from API
        poster_url = fetch_poster(movie_id)
        if poster_url is not None:
            recommended_movies_posters.append(poster_url)
        else:
            # Handle the case where fetch_poster returned None
            recommended_movies_posters.append("No Poster Available")

        # fetch movie details
        recommended_movies_details.append(format_movie_details(fetch_movie_details(movie_id)))

    return recommended_movies, recommended_movies_details, recommended_movies_posters


with open('movies_dict.pkl', 'rb') as file:
    movies_dict = pickle.load(file)

movies = pd.DataFrame.from_dict(movies_dict)

with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)


def summarizer_lex(text, num_sentences):
    # Use NLTK to tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Return the first 'num_sentences' sentences as the summary
    return sentences[:num_sentences]


def fetch_movie_details(movie_id):
    with requests.Session() as session:
        response = session.get(
            'https://api.themoviedb.org/3/movie/{}?api_key=b60699705806f6e441000719c0ef0ffb&language=en-US'.format(
                movie_id))
    data = response.json()
    movie_details = {
        'overview': data['overview'],
        'release_date': data['release_date'],
        'genres': [genre['name'] for genre in data['genres']],
        'runtime': data['runtime']
    }
    return movie_details


def format_movie_details(movie_details):
    summary = summarizer_lex(movie_details['overview'], 2)
    summary_text = ' '.join(summary)
    formatted_text = f"Overview:\n{summary_text}\n\n"
    formatted_text += f"Release Date: {movie_details['release_date']}\n\n"
    formatted_text += "Genres: {}\n\n".format(', '.join(movie_details['genres']))
    formatted_text += f"Runtime: {movie_details['runtime']} minutes\n\n"

    return formatted_text


def fetch_poster(movie_id):
    try:
        response = requests.get(
            'https://api.themoviedb.org/3/movie/{}?api_key=b60699705806f6e441000719c0ef0ffb&language=en-US'.format(
                movie_id), timeout=10)
        response.raise_for_status()
        data = response.json()

        # Add debugging statements to inspect the structure of the JSON response
        print(f"JSON Response: {data}")

        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            print(f"Poster path not found in the response for movie {movie_id}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie {movie_id}: {e}")
        return None


# WebPage
st.title('Movie Recommender System')

selected_movie_name = st.selectbox('Choose a movie to get recommendations', movies['title'].values)

if st.button('Recommend'):
    names, details, posters = recommend(selected_movie_name)

    row = st.columns(5)
    for i in range(5):
        with row[i]:
            st.header(names[i])
            st.image(posters[i])
            st.write(details[i], use_column_width=True)
