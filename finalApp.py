import pickle
import nltk
import pandas as pd
import requests
import streamlit as st
from langchain.adapters import openai
from langchain_community.llms import OpenAI
import datetime
import logging
import json
from PIL import Image

TMDB_API_KEY = "b60699705806f6e441000719c0ef0ffb"

# Streamlit setup
st.title('Movie Recommender System')

# Load movie data and similarity matrix
with open('movies_dict.pkl', 'rb') as file:
    movies_dict = pickle.load(file)

movies = pd.DataFrame.from_dict(movies_dict)

with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

# LangChain setup
langchain_api_key = "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"
llm = OpenAI(openai_api_key=langchain_api_key)


# Recommender function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    recommended_movies_details = []

    def summarizer_lex(text, num_sentences):
        # Use NLTK to tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)
        # Return the first 'num_sentences' sentences as the summary
        return sentences[:num_sentences]

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

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        # fetch poster from API
        poster_url = fetch_poster(movie_id)
        if poster_url is not None:
            recommended_movies_posters.append(poster_url)
        else:
            recommended_movies_posters.append("No Poster Available")
        # fetch movie details
        recommended_movies_details.append(format_movie_details(fetch_movie_details(movie_id)))

    return recommended_movies, recommended_movies_details, recommended_movies_posters


# Streamlit UI
selected_movie_name = st.selectbox('Choose a movie to get recommendations', movies['title'].values)

if st.button('Recommend'):
    names, details, posters = recommend(selected_movie_name)

    row = st.columns(5)
    for i in range(5):
        with row[i]:
            st.header(names[i])
            st.image(posters[i])
            st.write(details[i], use_column_width=True)

        # Get user preferences
        user_preference = st.text_input("Tell me about your movie preferences")


    def generate_query(text):
        # Define a dictionary of movie properties with descriptions
        movie_props = {f"movie_{i}": {"type": "string", "description": f"Movie {i}"} for i in range(1, 11)}

        # Create a message format for OpenAI Chat API
        message = [
            {"role": "system",
             "content": "You are a helpful assistant. Get each movie in the text. Movies should just be strings of "
                        "characters in lowercase, no number or date or parenthesis."},
            {"role": "user", "content": text},
        ]

        # Make a request to OpenAI's Chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=message,
            functions=[{
                "name": "get_movies",
                "description": "Returns 4 movies as the base search",
                "parameters": {
                    "type": "object",
                    "properties": movie_props,
                    "required": list(movie_props.keys()),
                },
            }],
            function_call={"name": "get_movies"},
            openai_api_key="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"
        )

        # Log the success of the OpenAI API request
        logging.info('OpenAI API request successful')

        # Extract information from the response
        message = response["choices"][0]["message"]
        movies = []

        # Check if the response contains a function call
        if message.get("function_call"):
            # Parse the arguments from the function call
            arguments = json.loads(message["function_call"]["arguments"])
            print(arguments)
            # Extract movies from the parsed arguments
            movies = [arguments[f"movie_{i}"] for i in range(1, 11)]

        return movies


    def search_movie(keywords):
        base_url = "https://api.themoviedb.org/3/search/movie"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {TMDB_API_KEY}"
        }

        results = []
        for keyword in keywords:
            params = {
                "query": keyword,
                "include_adult": "false",
                "language": "en-US",
                "page": 1
            }

            response = requests.get(base_url, headers=headers, params=params)

            if response.status_code == 200:
                logging.info(
                    'TMDB API request successful for keyword: %s', keyword)
                results.extend(response.json()['results'])
            else:
                logging.error('TMDB API request failed with status code: %s for keyword: %s. Details: %s',
                              response.status_code, keyword, response.text)

        results = process_movies(results)

        return results


    def process_movies(movies):
        current_year = datetime.datetime.now().year
        # Filter for movies from the last 3 years
        recent_movies = [movie for movie in movies if
                         movie['release_date'] and int(movie['release_date'][:4]) >= current_year - 10]
        # Remove duplicates
        unique_movies = {movie['id']: movie for movie in recent_movies}.values()
        # Sort by popularity
        sorted_movies = sorted(unique_movies, key=lambda m: m['popularity'], reverse=True)
        # Return the top 21, or less if there aren't that many
        return sorted_movies[:21]


    # Call Chat API to generate more movie suggestions based on user preferences
    prompt = f"I watched {', '.join(names)}. Based on this, my preferences are: {user_preference}. Can you suggest more movies?"
    logging.info('Getting additional movie suggestions from OpenAI with prompt: %s', prompt)
    additional_movie_gpt = llm.generate([prompt]).generations[0][0].text
    logging.info('Received additional movie suggestions from OpenAI: %s', additional_movie_gpt)


    #print statement to check logs
    st.write("" + additional_movie_gpt);

    # Extract movies from the LangChain response
    additional_movie_query = generate_query(additional_movie_gpt)
    logging.info('Received additional movies keywords from OpenAI: %s', additional_movie_query)

    # Search for additional movies using TMDb API
    additional_suggested_movies = search_movie(additional_movie_query)

    # Display additional suggested movies
    if additional_suggested_movies:
        st.write("Here are more movies you might like based on your preferences:")
        for movie in additional_suggested_movies:
            st.write("- " + movie)
    else:
        st.write("Sorry, I couldn't find any more movies that match your preferences.")
