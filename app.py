import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"



# Update the random_image_urls list with just filenames
random_image_urls = [
    "img_1.png", "img_2.png", "img_3.png", "img_4.png",
    "img_5.png", "img_6.png", "img_7.png", "img_8.png",
]

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
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']
    ]
    return recommended_items_details


random_prices = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

# CSS Styling
st.markdown("""
    <style>
        .main-container {
            padding: 2rem;
        }
        .product-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1.5rem;
            padding: 2rem 0;
        }
        .product-card {
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
            display: flex;
            gap: 1rem;
            margin-bottom : 20px;
        }
        .product-image-container {
            flex: 1;
        }
        .product-details {
            flex: 2;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .product-image {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .product-name {
            font-size: 1rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .product-info {
            font-size: 0.9rem;
            color: #666;
            margin: 0.25rem 0;
        }
        .rating {
            color: #f1c40f;
        }
    </style>
""", unsafe_allow_html=True)

# App UI
st.title("E-Commerce Product Recommendation System 🚀")
st.write("Select a product to view similar recommendations based on content.")

# Product Selection
product_options = data['Name'].dropna().unique().tolist()
item_name_input = st.selectbox("Choose a Product", product_options)

# Get Recommendations
if st.button("Get Recommendations ➡️"):
    if item_name_input:
        recommendations = content_based_recommendations(data, item_name_input, top_n=10)
        
        if recommendations.empty:
            st.error("No recommendations found 😞")
        else:
            st.write("### Recommended Items")
            st.markdown('<div class="product-container">', unsafe_allow_html=True)
            
            # Replace the recommendations loop with this:
            for idx, row in recommendations.iterrows():

                random_image = random.choice(random_image_urls)
                image_path = f"./static/{random_image}"
                base64_img = get_base64_image(image_path)

                st.markdown(f"""
                    <div class="product-card">
                        <div class="product-image-container">
                            <img src="{base64_img}" class="product-image" alt="Product Image">
                        </div>
                        <div class="product-details">
                            <div class="product-name">{row['Name']}</div>
                            <div class="product-info">Brand: {row['Brand']}</div>
                            <div class="product-info">Price: ${random.choice(random_prices):.2f}</div>
                            <div class="product-info">Rating: <span class="rating">{'⭐' * int(row['Rating'])}</span></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                print("hello");

                        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Please select a product!")