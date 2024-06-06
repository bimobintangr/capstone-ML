import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

css_kustom = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Monofett&display=swap');
.stApp {
    background-color: #FAE392;
}
.title-center {
    text-align: center;
    color: black;  /* Ubah warna teks jika perlu */
    font-family: 'Monofett', cursive;
    font-size: 96px; /* Ubah ukuran font sesuai kebutuhan */
    line-height: 1,5; /* Ubah line height sesuai kebutuhan */
}
.title-center1 {
    text-align: center;
    color: black;  /* Ubah warna teks jika perlu */
}
.stButton button {
    height: 45px;
    background-color: #3e873d;
    color: #f9d14a;
    border: none;
    border-radius: 20px;
    font-family: 'Ibarra Real Nova', cursive;
    font-size: 24px;
    gap: 0px;
    opacity: 1;
    
}
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)
# Load dataset
df = pd.read_csv('Dataset Rekomendasi Makanan Favorit.csv')

# Data preprocessing
features = ["Suka Pedas", "Suka Manis", "Suka Asin", "Suka Asam", "Suka Makanan Berkuah", "Jenis Kelamin", "Umur"]
X = pd.get_dummies(df[features], drop_first=True)

# Menggunakan LabelEncoder untuk mengubah label menjadi integer
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
df["Prioritas 1"] = le1.fit_transform(df["Prioritas 1"])
df["Prioritas 2"] = le2.fit_transform(df["Prioritas 2"])
df["Prioritas 3"] = le3.fit_transform(df["Prioritas 3"])

y = df[["Prioritas 1", "Prioritas 2", "Prioritas 3"]].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(le1, 'label_encoder_1.pkl')
joblib.dump(le2, 'label_encoder_2.pkl')
joblib.dump(le3, 'label_encoder_3.pkl')


# Daftar pengguna dan password (hardcoded untuk contoh ini)
users = {
    "admin": "admin123",
    "user1": "password1",
    "user2": "password2"
}

# Fungsi untuk mengecek login
def check_login(username, password):
    if username in users and users[username] == password:
        return True
    return False

# Inisialisasi session state untuk halaman dan status login
if 'halaman' not in st.session_state:
    st.session_state.halaman = "halaman_utama"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Load the model and label encoders
model = joblib.load('random_forest_model.pkl')
le1 = joblib.load('label_encoder_1.pkl')
le2 = joblib.load('label_encoder_2.pkl')
le3 = joblib.load('label_encoder_3.pkl')

# Fungsi untuk halaman utama
def halaman_utama():
   
    st.markdown('<h1 class="title-center">REMAFA</h1>', unsafe_allow_html=True)
    image_paths = ["bibimbap.png"]
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image_paths[0], use_column_width=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("Memulai", use_container_width=True):
            st.session_state.halaman = "login_page"

# Fungsi untuk halaman login
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.halaman = "halaman_2"
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password")

# Fungsi untuk halaman input preferensi
def halaman_2():
    st.markdown('<h1 class="title-center1">APLIKASI REKOMENDASI MAKANAN NUSANTARA SESUAI SELERA</h1>', unsafe_allow_html=True)
    pedas = st.selectbox("Apakah kamu suka pedas?",["Iya", "Tidak"])
    manis = st.selectbox("Apakah kamu suka Manis?",["Iya", "Tidak"])
    asin = st.selectbox("Apakah kamu suka Asin?",["Iya", "Tidak"])
    asam = st.selectbox("Apakah kamu suka Asam?",["Iya", "Tidak"])
    kuah = st.selectbox("Apakah kamu suka makanan berkuah?",["Iya", "Tidak"])
    jenis_kelamin = st.selectbox("Jenis Kelamin",["Laki-laki", "Wanita"])
    age = st.number_input("Umur", min_value=0, max_value=100, value=0)
    
    if st.button("Cari"):
        input_data = np.array([
            1 if pedas == "Iya" else 0,
            1 if manis == "Iya" else 0,
            1 if asin == "Iya" else 0,
            1 if asam == "Iya" else 0,
            1 if kuah == "Iya" else 0,
            1 if jenis_kelamin == "Laki-laki" else 0,
            age
        ]).reshape(1, -1)
        
        # Predict using the Random Forest model
        predictions = model.predict(input_data)
        
        # Decode the predictions back to food names
        pred_1 = le1.inverse_transform([predictions[0][0]])[0]
        pred_2 = le2.inverse_transform([predictions[0][1]])[0]
        pred_3 = le3.inverse_transform([predictions[0][2]])[0]
        
        st.session_state.recommendations = [pred_1, pred_2, pred_3]
        st.session_state.halaman = "halaman_3"

# Fungsi untuk halaman rekomendasi
def halaman_3():
    st.markdown('<h1 class="title-center1">APLIKASI REKOMENDASI MAKANAN NUSANTARA SESUAI SELERA</h1>', unsafe_allow_html=True)
    image_paths = ["Ayam Geprek.png", "Bakso.png", "Bubur.png", "Getuk.png", "Lamongan.png", "Mie Ayam.png", "Nasi Kuning.png", "Nasi Padang.png", "Nasi Uduk.png", "Sate.png", "Sayur Sop.png", "Seafood.png", "Seblak.png", "Soto Ayam.png", "Tongseng.png"]
    captions = ["Ayam Geprek", "Bakso", "Bubur", "Getuk", "Lamongan", "Mie Ayam", "Nasi Kuning", "Nasi Padang", "Nasi Uduk", "Sate", "Sayur Sop", "Seafood", "Seblak", "Soto Ayam", "Tongseng"]
    
    # Map food names to indices
    food_to_index = {food: i for i, food in enumerate(captions)}
    
    # Get indices of recommendations
    rec_indices = [food_to_index[food] for food in st.session_state.recommendations]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image_paths[rec_indices[0]], use_column_width=True)
        st.markdown(f"**{captions[rec_indices[0]]}**")
    with col2:
        st.image(image_paths[rec_indices[1]], use_column_width=True)
        st.markdown(f"**{captions[rec_indices[1]]}**")
    with col3:
        st.image(image_paths[rec_indices[2]], use_column_width=True)
        st.markdown(f"**{captions[rec_indices[2]]}**")
    
    if st.button("Kembali"):
        st.session_state.halaman = "halaman_2"

# Navigasi berdasarkan state halaman
if st.session_state.halaman == "halaman_utama":
    halaman_utama()
elif st.session_state.halaman == "login_page":
    login_page()
elif st.session_state.halaman == "halaman_2":
    if st.session_state.logged_in:
        halaman_2()
    else:
        st.warning("Silakan login terlebih dahulu.")
        login_page()
elif st.session_state.halaman == "halaman_3":
    if st.session_state.logged_in:
        halaman_3()
    else:
        st.warning("Silakan login terlebih dahulu.")
        login_page()
