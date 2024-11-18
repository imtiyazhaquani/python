
import pickle
import streamlit as st
import numpy as np

# Loading Required Data
model = pickle.load(open('/Users/imtiyazhaquani/Downloads/Book_Recommendation_System-master 3/artifacts/model.pkl','rb'))
book_names = pickle.load(open('/Users/imtiyazhaquani/Downloads/Book_Recommendation_System-master 3/artifacts/book_names.pkl','rb'))
filtered_data = pickle.load(open('/Users/imtiyazhaquani/Downloads/Book_Recommendation_System-master 3/artifacts/filtered_data.pkl','rb'))
filtered_data_pivot_table = pickle.load(open('/Users/imtiyazhaquani/Downloads/Book_Recommendation_System-master 3/artifacts/filtered_data_pivot_table.pkl','rb'))

# Defining the Function to Fetch Book Poster URLs
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(filtered_data_pivot_table.index[book_id])

    for name in book_name[0]: 
        ids = np.where(filtered_data['Book-Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = filtered_data.iloc[idx]['Image-URL-M']
        poster_url.append(url)

    return poster_url


# Defining the Book Recommendation Function
def recommend_book(book_name):
    books_list = []
    book_id = np.where(filtered_data_pivot_table.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(filtered_data_pivot_table.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
            books = filtered_data_pivot_table.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    return books_list , poster_url       


# Creating the Streamlit UI
st.header('Book Recommender System')
selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names)

# Displaying the Recommendations
if st.button('Show Recommendation'):
    recommended_books,poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])