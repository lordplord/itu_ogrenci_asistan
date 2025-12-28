import os
import time
import pandas as pd
from groq import Groq
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. AYARLAR
# ==========================================
# Buraya console.groq.com'dan aldığın anahtarı yapıştır
GROQ_API_KEY = "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" 

CSV_FILE = "ogrenci_isleri_veri_seti_1000.csv"
TEST_SAMPLE_SIZE = 50  # Test edilecek soru sayısı

# İstemciyi başlat
client = Groq(api_key=GROQ_API_KEY)

# Llama 3 modelini kullanıyoruz (Çok hızlı ve zekidir)
MODEL_NAME = "llama-3.3-70b-versatile" 

# ==========================================
# 2. FONKSİYONLAR
# ==========================================

def get_prediction_groq(text):
    # Prompt Mühendisliği: Modele örnekler (Few-Shot) veriyoruz
    system_prompt = """
    Sen İTÜ Öğrenci İşleri için çalışan uzman bir yapay zeka asistanısın.
    Görevin: Gelen mesajı analiz et ve aşağıdaki 5 etiketlen SADECE birini seçip yaz.
    
    ETİKET LİSTESİ:
    - ders_kaydi
    - transkript
    - yatay_gecis
    - selamlama
    - vedalasma
    
    ÖRNEKLER (Buna göre karar ver):
    Kullanıcı: "Merhaba kolay gelsin" -> Çıktı: selamlama
    Kullanıcı: "Not dökümümü nasıl alırım?" -> Çıktı: transkript
    Kullanıcı: "Ders seçimi ne zaman başlıyor?" -> Çıktı: ders_kaydi
    Kullanıcı: "Başka bölüme geçmek istiyorum şartlar ne?" -> Çıktı: yatay_gecis
    Kullanıcı: "Teşekkürler iyi günler" -> Çıktı: vedalasma
    
    KURAL: Sadece tek bir kelime (etiket) yaz. Açıklama yapma.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model=MODEL_NAME,
            temperature=0, # Yaratıcılığı sıfırla, robot gibi net olsun
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Hata: {e}")
        return "error"

# ==========================================
# 3. VERİ YÜKLEME VE TEST
# ==========================================
print(f"Veri seti yükleniyor: {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print("CSV dosyası bulunamadı!")
    exit()

# Veriyi karıştır ve test kümesini ayır
df = df.sample(frac=1).reset_index(drop=True)
test_df = df.head(TEST_SAMPLE_SIZE).copy()

print(f"\n🚀 Groq (Llama 3) ile {TEST_SAMPLE_SIZE} adet veri test ediliyor...\n")

y_true = []
y_pred = []

baslangic = time.time()

for index, row in test_df.iterrows():
    text = row['text']
    actual_intent = row['intent'].strip()
    
    # Groq'a sor
    prediction = get_prediction_groq(text)
    
    # Bazen model "Etiket: ders_kaydi" diyebilir, temizleyelim
    # (Llama 3 genelde söz dinler ama önlem alalım)
    if ":" in prediction:
        prediction = prediction.split(":")[-1].strip()
        
    y_true.append(actual_intent)
    y_pred.append(prediction)
    
    print(f"[{index+1}/{TEST_SAMPLE_SIZE}] Soru: {text[:30]}... -> Tahmin: {prediction}")
    
    # Groq çok hızlıdır ama yine de nezaketen minik bir bekleme koyalım
    # Dakikada 30 isteğe kadar izin verir.
    time.sleep(3) 

bitis = time.time()
print(f"\nTest tamamlandı! Geçen süre: {bitis - baslangic:.2f} saniye")

# ==========================================
# 4. RAPORLAMA
# ==========================================
print("\n" + "="*50)
print("GROQ - LLAMA 3 SINIFLANDIRMA RAPORU")
print("="*50)

# Metrikler
print(classification_report(y_true, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
labels = sorted(list(set(y_true + y_pred)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
plt.title(f'Confusion Matrix (Model: {MODEL_NAME})')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()