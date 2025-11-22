# ===========================================================
# 📚 BOOK RECOMMENDATION SYSTEM USING KNN (STREAMLIT APP)
# ===========================================================
# Autor: Ricardo Leite
# Propósito: Sistema simples de recomendação baseado em similaridade
# Dataset: Book-Crossings (1.1M ratings)
# Framework Web: Streamlit
# ===========================================================
⚠️ Os arquivos CSV originais não são enviados ao GitHub por terem tamanho elevado.
Utilize os links abaixo para baixar os dados:

wget https://cdn.freecodecamp.org/project-data/book-recommendation-system/BX-Books.csv
wget https://cdn.freecodecamp.org/project-data/book-recommendation-system/BX-Book-Ratings.csv

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import time


# ===========================================================
# ⚙️ CONFIGURAÇÃO DA INTERFACE
# ===========================================================
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation System using KNN")
st.markdown("Recomende automaticamente livros similares com base em avaliações reais de usuários.")


# ===========================================================
# 📥 CARREGAMENTO E CACHE DE DADOS
# ===========================================================
@st.cache_resource
def load_data():
    """ Carrega os arquivos CSV apenas uma vez (cache) """
    try:
        df_books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', low_memory=False)
        df_ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', low_memory=False)
        return df_books, df_ratings
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None


# ===========================================================
# 🧹 PRÉ-PROCESSAMENTO (LIMPEZA)
# ===========================================================
def preprocess_data(df_books, df_ratings):
    """ Limpa dados e cria matriz livro x usuário """
    st.info("🔄 Processando dados...")

    df_books = df_books.dropna(subset=['title'])

    user_counts = df_ratings['user_id'].value_counts()
    df_ratings = df_ratings[df_ratings['user_id'].isin(user_counts[user_counts >= 200].index)]

    book_counts = df_ratings['isbn'].value_counts()
    df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]

    matrix = df_ratings.pivot_table(index='isbn', columns='user_id', values='rating').fillna(0)
    sparse_matrix = csr_matrix(matrix.values)

    return df_books, matrix, sparse_matrix


# ===========================================================
# 🤖 TREINAMENTO DO MODELO
# ===========================================================
def train_model(sparse_matrix):
    """ Treina modelo KNN """
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)
    return model


# ===========================================================
# 🔍 FUNÇÃO PRINCIPAL DE RECOMENDAÇÃO
# ===========================================================
def get_recommends(title, df_books, matrix, sparse_matrix, model):
    """ Retorna top 5 livros similares ao título """
    try:
        isbn = df_books[df_books['title'] == title]['isbn'].values[0]
    except IndexError:
        return "❌ Livro não encontrado no dataset."

    try:
        index = matrix.index.tolist().index(isbn)
    except ValueError:
        return "⚠ Livro encontrado, mas sem avaliações suficientes."

    distances, indices = model.kneighbors(sparse_matrix[index], n_neighbors=6)
    recommendations = []

    for i in range(1, len(indices[0])):
        book_isbn = matrix.index[indices[0][i]]
        rec_title = df_books[df_books['isbn'] == book_isbn]['title'].values[0]
        recommendations.append((rec_title, round(distances[0][i], 4)))

    return recommendations


# ===========================================================
# 🧪 EXECUÇÃO PRINCIPAL DO APP
# ===========================================================
df_books, df_ratings = load_data()

if df_books is None:
    st.stop()

df_books, matrix, sparse_matrix = preprocess_data(df_books, df_ratings)
model = train_model(sparse_matrix)


# ===========================================================
# 🧠 BARRA LATERAL DE AJUDA
# ===========================================================
with st.sidebar:
    st.header("ℹ️ Como usar")
    st.write("""
    1. Digite o nome exato de um livro.
    2. Clique em **Recomendar**.
    3. Receba 5 livros similares via KNN.
    \n
    *Exemplo real:*  
    `"The Queen of the Damned (Vampire Chronicles (Paperback))"`
    """)

    if st.button("Ver livros disponíveis"):
        st.write(df_books['title'].sample(10).tolist())


# ===========================================================
# 🔎 CAMPO DE BUSCA
# ===========================================================
st.subheader("🔎 Buscar recomendações")
book_title = st.text_input("Digite o nome exato do livro:")

if st.button("Recomendar"):
    with st.spinner("Calculando recomendações..."):
        time.sleep(1)
        recs = get_recommends(book_title, df_books, matrix, sparse_matrix, model)

    if isinstance(recs, list):
        st.success("📚 Recomendações encontradas!")
        st.table(pd.DataFrame(recs, columns=["Livro Similar", "Distância"]))
    else:
        st.error(recs)
