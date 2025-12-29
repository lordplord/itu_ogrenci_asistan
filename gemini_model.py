import google.generativeai as genai
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. AYARLAR (BURAYI DÜZENLE)
# ==========================================
API_KEY = "*************************"  # <-- Aldığın anahtarı tırnak içine yapıştır
CSV_FILE = "ogrenci_isleri_veri_seti_1000.csv" # Senin oluşturduğun CSV dosyasının adı
TEST_SAMPLE_SIZE = 50  # Test edilecek soru sayısı (Hızlı sonuç için 50 ideal)

# ==========================================
# 2. MODEL HAZIRLIĞI
# ==========================================
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Modelin görevi (Prompt)
system_instruction = """
Sen bir öğrenci işleri asistanısın. Aşağıdaki metni analiz et ve sadece şu etiketlerden birini döndür:
'ders_kaydi', 'transkript', 'yatay_gecis', 'selamlama', 'vedalasma'.
Başka hiçbir açıklama yapma, sadece etiketi yaz.
"""

def get_prediction(text):
    try:
        # Modele soruyu sor
        response = model.generate_content(f"{system_instruction}\n\nSoru: {text}")
        # Cevabı temizle (boşlukları at, küçük harf yap)
        return response.text.strip().lower()
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return "error"

# ==========================================
# 3. VERİ YÜKLEME VE TEST ETME
# ==========================================
print(f"Dosya okunuyor: {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print("HATA: CSV dosyası bulunamadı! Dosya adının doğru olduğundan emin ol.")
    exit()

# Veriyi karıştır ve test için ayır
df = df.sample(frac=1).reset_index(drop=True)
test_df = df.head(TEST_SAMPLE_SIZE).copy() # İlk 50 tanesini test et

print(f"\nToplam {TEST_SAMPLE_SIZE} soru test ediliyor, lütfen bekle...\n")

y_true = [] # Gerçek etiketler
y_pred = [] # Modelin tahminleri

for index, row in test_df.iterrows():
    text = row['text']
    actual_intent = row['intent'].strip().lower() # CSV'deki gerçek cevap
    
    # Gemini'ye sor
    predicted_intent = get_prediction(text)
    
    # Listelere ekle
    y_true.append(actual_intent)
    y_pred.append(predicted_intent)
    
    # Ekrana yazdır (İlerlemeyi görmek için)
    print(f"Soru: {text[:40]}... -> Tahmin: {predicted_intent} | Gerçek: {actual_intent}")
    
    time.sleep(12) # API'yi yormamak için kısa bekleme

# ==========================================
# 4. SONUÇ RAPORU (METRİKLER)
# ==========================================
print("\n" + "="*50)
print("DEĞERLENDİRME RAPORU")
print("="*50)

# Precision, Recall, F1 Score Tablosu
print(classification_report(y_true, y_pred, zero_division=0))

# Confusion Matrix (Görsel Grafik)
print("Confusion Matrix çiziliyor...")
cm = confusion_matrix(y_true, y_pred)
labels = sorted(list(set(y_true + y_pred))) # Tüm etiketleri bul

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Tahmin Edilen (Predicted)')
plt.ylabel('Gerçek Olan (Actual)')
plt.title('Confusion Matrix (Hata Matrisi)')

plt.show()
