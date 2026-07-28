import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px


st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew"
        ]
    ]

    movies.dropna(inplace=True)

    return movies


def convert(text):

    result = []

    for item in json.loads(text):
        result.append(item["name"])

    return result


def convert_cast(text):

    result = []

    for i, item in enumerate(json.loads(text)):

        if i < 3:
            result.append(item["name"])
        else:
            break

    return result


def fetch_director(text):

    for item in json.loads(text):

        if item["job"] == "Director":
            return [item["name"]]

    return []


@st.cache_data
def preprocess():

    movies = load_data()

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(
        lambda x: x.split()
    )

    for col in ["genres", "keywords", "cast", "crew"]:

        movies[col] = movies[col].apply(
            lambda x: [i.replace(" ", "") for i in x]
        )

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[
        ["movie_id", "title", "tags"]
    ].copy()

    new_df["tags"] = new_df["tags"].apply(
        lambda x: " ".join(x).lower()
    )

    return new_df


@st.cache_data
def build_similarity():

    new_df = preprocess()

    # Collect all words
    all_words = []

    for text in new_df["tags"]:
        all_words.extend(text.split())

    # Unique words
    unique_words = np.unique(all_words)

    # Limit vocabulary size
    unique_words = unique_words[:5000]

    vocab = {
        word: idx
        for idx, word in enumerate(unique_words)
    }

    vectors = np.zeros(
        (len(new_df), len(vocab)),
        dtype=np.uint16
    )

    for row_idx, text in enumerate(new_df["tags"]):

        for word in text.split():

            if word in vocab:
                vectors[row_idx][vocab[word]] += 1

    # Cosine Similarity
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    normalized_vectors = vectors / norms

    similarity = np.dot(
        normalized_vectors,
        normalized_vectors.T
    )

    return new_df, similarity


with st.spinner("Preparing recommendation engine..."):
    new_df, similarity = build_similarity()


def recommend(movie_name):

    movie_index = new_df[
        new_df["title"] == movie_name
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    scores = []

    for movie in movie_list:

        names.append(
            new_df.iloc[movie[0]].title
        )

        scores.append(
            round(float(movie[1] * 100), 2)
        )

    return names, scores


st.title(" Content-Based Movie Recommendation Engine")

st.write(
    "Discover similar movies using NumPy-based cosine similarity."
)

selected_movie = st.selectbox(
    "Choose a Movie",
    sorted(new_df["title"].unique())
)

if st.button("Recommend Movies"):

    names, scores = recommend(selected_movie)

    st.subheader("Top 5 Similar Movies")


    chart_df = pd.DataFrame(
        {
            "Movie": names,
            "Similarity Score": scores
        }
    )

    fig = px.bar(
        chart_df,
        x="Similarity Score",
        y="Movie",
        orientation="h",
        text="Similarity Score",
        title=f"Movies Similar to '{selected_movie}'"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )