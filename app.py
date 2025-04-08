import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Loading
@st.cache_data
def load_data():
    data = pd.read_csv("data/clean_data.csv")
    return data

data = load_data()

# Content Based Recommendation
def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        st.error(f"❌ Item '{item_name}' not found.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on their TF-IDF vectors
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the input item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Enumerate similarity scores with all other items
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Return the details of the recommended items
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ]
    return recommended_items_details

# Dummy images and prices
random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png",
]

random_prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

# CSS Styling for spacing and alignment
st.markdown("""
    <style>
        # .product-card {
        #     background-color: #f9f9f9;
        #     border-radius: 10px;
        #     padding: 15px;
        #     margin: 10px;
        #     text-align: center;
        #     height: 100%;
        #     box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        # }
        .product-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App UI
st.title("E-Commerce Product Recommendation System 🚀")
st.write("Select a product item to view the top 10 recommended products based on similar content.")

# Dropdown instead of text input
product_options = data['Name'].dropna().unique().tolist()
item_name_input = st.selectbox("Choose a Product", product_options)

# Recommendation Button
if st.button("Get Recommendations ➡️"):
    if item_name_input:
        recommendations = content_based_recommendations(data, item_name_input, top_n=10)
        if recommendations.empty:
            st.write("No recommendations found 😞")
        else:
            st.write("### Recommended Items")
            cols = st.columns(5)  # Flex-style layout: 5 items per row
            for idx, row in enumerate(recommendations.iterrows()):
                col = cols[idx % 5]
                with col:
                    image_url = random.choice(random_image_urls)
                    price = random.choice(random_prices)
                    with st.container():
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)
                        st.image(image_url, use_container_width=True)
                        st.markdown(f"**{row[1]['Name']}**")
                        st.markdown(f"**Brand:** {row[1]['Brand']}")
                        st.markdown(f"**Price:** ${price:.2f}")
                        st.markdown(f"**Rating:** {row[1]['Rating']} ⭐")
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Please select a product item!")
