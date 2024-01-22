import datetime
import requests
import logging
import json
import openai
import streamlit as st
from PIL import Image
from langchain_community.llms import OpenAI

TMDB_API_KEY = "b60699705806f6e441000719c0ef0ffb"

logging.basicConfig(level=logging.INFO)
OpenAI.api_key = "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"



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
    current_year = datetime.now().year
    # Filter for movies from the last 3 years
    recent_movies = [movie for movie in movies if
                     movie['release_date'] and int(movie['release_date'][:4]) >= current_year - 10]
    # Remove duplicates
    unique_movies = {movie['id']: movie for movie in recent_movies}.values()
    # Sort by popularity
    sorted_movies = sorted(unique_movies, key=lambda m: m['popularity'], reverse=True)
    # Return the top 21, or less if there aren't that many
    return sorted_movies[:21]


def converse_with_langchain(prompt):
    llm = OpenAI(
        openai_api_key="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"
    )

    llm_result = llm.generate([prompt])

    # We're assuming that the first response is the one we want
    # This might need to be adjusted depending on your specific use case
    response = llm_result.generations[0][0].text

    logging.info('LangChain API request successful')

    # Return the assistant's reply
    return response


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


def get_movie_suggestion(user_preference):
    # Here we generate a prompt to ask GPT-3 about movie
    # suggestions based on user preferences
    prompt = f"""I am looking for a movie. My preferences are:
                    {user_preference}. Can you suggest something? (10 movies)"""
    logging.info('Getting movie suggestion from OpenAI with prompt: %s', prompt)

    movie_gpt = converse_with_langchain(prompt)
    logging.info('Received movie suggestion from OpenAI: %s', movie_gpt)

    movie_query = generate_query(movie_gpt)

    logging.info('Received movies keywords from OpenAI: %s', movie_query)

    return search_movie(movie_query)


def get_watch_providers(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return None


st.title('Movie Recommendation Bot')

user_preference = st.text_input("Tell me about your movie preferences")

# Define the base URL for images
image_base_url = "https://image.tmdb.org/t/p/w500"

if st.button('Get Movie Recommendation'):
    st.spinner('Searching for your movie...')
    movies = get_movie_suggestion(user_preference)

    if movies:
        logging.info('Found movie(s) matching the query')
        st.write("Here are some movies you might like:")

        # Initialize column counter
        col_count = 0

        for movie in movies:
            # Start a new row every 3 movies
            if col_count % 3 == 0:
                cols = st.columns(3)

            # Show the movie poster image in the appropriate column
            with cols[col_count % 3]:
                if movie['poster_path']:
                    # Get image
                    img_url = image_base_url + movie['poster_path']
                    response = requests.get(img_url, stream=True)
                    img = Image.open(response.raw)

                    # Adjust image size
                    baseheight = 300
                    hpercent = (baseheight / float(img.size[1]))
                    wsize = int((float(img.size[0]) * float(hpercent)))
                    img = img.resize((wsize, baseheight), Image.LANCZOS)

                    # Display the image
                    st.image(img)

                    # Display the movie information
                    st.write("Title: " + movie['title'])
                    st.write("Popularity: " + str(movie['popularity']))
                    st.write("Vote Average: " + str(movie['vote_average']))
                    st.write("Vote Count: " + str(movie['vote_count']))

                    # Fetch and display the watch providers
                    providers = get_watch_providers(movie['id'])
                    # Fetch and display the watch providers
                    providers = get_watch_providers(movie['id'])
                    if providers and 'results' in providers:
                        providers_list = []
                        for country_code, country_providers in providers['results'].items():
                            for service in ['flatrate', 'rent', 'buy']:
                                if service in country_providers:
                                    providers_list.extend(
                                        [provider['provider_name'] for provider in country_providers[service]])

                        # Remove duplicate providers by converting the list to a set and back to a list
                        providers_list = list(set(providers_list))

                        with st.expander('Available Watch Providers'):
                            st.write(', '.join(providers_list))


                else:
                    st.write("No image available for this movie")

            # Increment column counter
            col_count += 1

    else:
        logging.warning('No movies found matching the query')
        st.write("""Sorry, I couldn't find any movies that match your preferences.""")
