
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

@st.cache_resource
def load_data():
    df_books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', low_memory=False)
    df_ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', low_memory=False)
    return df_books, df_ratings

def preprocess_data(df_books, df_ratings):
    df_books = df_books.dropna(subset=['title'])
    user_counts = df_ratings['user_id'].value_counts()
    df_ratings = df_ratings[df_ratings['user_id'].isin(user_counts[user_counts >= 200].index)]
    book_counts = df_ratings['isbn'].value_counts()
    df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]
    matrix = df_ratings.pivot_table(index='isbn', columns='user_id', values='rating').fillna(0)
    sparse_matrix = csr_matrix(matrix.values)
    return df_books, matrix, sparse_matrix

def train_model(sparse_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)
    return model

def get_recommends(title, df_books, matrix, sparse_matrix, model):
    try:
        isbn = df_books[df_books['title'] == title]['isbn'].values[0]
    except IndexError:
        return f"Livro '{title}' não encontrado."

    try:
        index = matrix.index.tolist().index(isbn)
    except ValueError:
        return "Livro encontrado, mas sem avaliações suficientes."

    distances, indices = model.kneighbors(sparse_matrix[index], n_neighbors=6)
    recommendations = []
    for i in range(1, len(indices[0])):
        book_isbn = matrix.index[indices[0][i]]
        rec_title = df_books[df_books['isbn'] == book_isbn]['title'].values[0]
        recommendations.append((rec_title, distances[0][i]))
    return recommendations

st.title("📚 Book Recommendation System")
df_books, df_ratings = load_data()
df_books, matrix, sparse_matrix = preprocess_data(df_books, df_ratings)
model = train_model(sparse_matrix)

book_title = st.text_input("Digite o nome do livro:")
if st.button("Recomendar"):
    recs = get_recommends(book_title, df_books, matrix, sparse_matrix, model)
    st.write(recs)
