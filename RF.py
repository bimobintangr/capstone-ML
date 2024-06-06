import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'Dataset Rekomendasi Makanan Favorit.csv'
dataset = pd.read_csv(file_path)

# Preprocessing dataset
def convert_income_to_numeric(income):
    if 'sampai' in income:
        return sum(map(int, income.replace('.', '').split(' sampai '))) / 2
    elif 'Belum ada pendapatan' in income:
        return 0
    elif '> 20000000' in income:
        return 20000000
    else:
        return int(income.replace('.', '').replace('>', '').strip())

def convert_spending_to_numeric(spending):
    return int(spending.replace('.', '').replace(',', '').replace('Rp', '').strip())

dataset['Pendapatan Perbulan'] = dataset['Pendapatan Perbulan'].apply(convert_income_to_numeric)
dataset['Berapa uang yang anda keluarkan untuk membeli makanan dalam sehari?'] = dataset['Berapa uang yang anda keluarkan untuk membeli makanan dalam sehari?'].apply(convert_spending_to_numeric)

categorical_columns = [
    'Jenis Kelamin', 'Status', 'Pendidikan Terakhir', 'Bidang Pekerjaan', 'Provinsi', 'Kabupaten/Kota',
    'Apakah Anda Sering Liburan', 'Apakah Anda Hobi Kuliner', 'Suka Pedas', 'Suka Manis', 'Suka Asin',
    'Suka Asam', 'Suka Makanan Berkuah', 'Prioritas 1', 'Prioritas 2', 'Prioritas 3'
]

dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns)

numerical_columns = ['Umur', 'Pendapatan Perbulan', 'Berapa uang yang anda keluarkan untuk membeli makanan dalam sehari?', 'Berapa kali anda membeli makanan keluar dalam seminggu?']

scaler = StandardScaler()
dataset_encoded[numerical_columns] = scaler.fit_transform(dataset_encoded[numerical_columns])

# Assume target column 'Prioritas 1' for this example
X = dataset_encoded.drop(columns=['Prioritas 1_Seafood', 'Prioritas 1_Lamongan', 'Prioritas 1_Nasi Padang', 'Prioritas 1_Soto Ayam', 'Prioritas 1_Ayam Geprek'])
y = dataset_encoded[['Prioritas 1_Seafood', 'Prioritas 1_Lamongan', 'Prioritas 1_Nasi Padang', 'Prioritas 1_Soto Ayam', 'Prioritas 1_Ayam Geprek']].idxmax(axis=1)

# Menggunakan SMOTE untuk menyeimbangkan dataset
smote = SMOTE(random_state=42)
y_numerical = y.astype('category').cat.codes
X_resampled, y_resampled = smote.fit_resample(X, y_numerical)

# Membagi ulang data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Melatih kembali model Random Forest dengan data yang sudah diseimbangkan
params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1}
rf_model = RandomForestClassifier(**params, random_state=42)
rf_model.fit(X_train, y_train)

# Memprediksi hasil pada data uji
y_pred = rf_model.predict(X_test)

# Menghitung akurasi dan laporan klasifikasi
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=y.astype('category').cat.categories)

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
encoder_columns = dataset_encoded.columns
joblib.dump(encoder_columns, 'encoder_columns.pkl')

print("Accuracy:", accuracy)
print("Classification Report:", report)
