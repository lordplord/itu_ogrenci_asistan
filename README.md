# Proje Özeti: Üretken Yapay Zeka Destekli İTÜ Öğrenci İşleri Asistanı

## Proje Tanımı
Bu proje, İTÜ öğrencilerinin sıkça sorduğu akademik ve idari soruların (ders kaydı, transkript, yatay geçiş vb.) 7/24 yanıtlanması amacıyla geliştirilmiş, **Büyük Dil Modelleri (LLM)** tabanlı interaktif bir sohbet botudur.

Proje, sadece metin üretmekle kalmayıp, kullanıcının niyetini (intent) anlayarak spesifik ve doğrulanmış bilgiler sunan hibrit bir mimariye sahiptir.

## Amaç
Öğrenci işleri birimlerindeki tekrarlayan soru yükünü hafifletmek, öğrencilere anlık geri bildirim sağlamak ve "halüsinasyon" (yanlış bilgi üretme) riskini minimize ederek güvenilir bir dijital asistan oluşturmak.

## Kullanılan Yöntem ve Mimari
Sistem, **"Önce Anla, Sonra Yanıtla"** prensibiyle çalışan iki aşamalı bir boru hattı (pipeline) üzerine kurulmuştur:

1.  **Niyet Sınıflandırma (Intent Classification):** Kullanıcıdan gelen metin, arka planda çalışan LLM tarafından analiz edilerek önceden tanımlı kategorilere (Örn: `ders_kaydi`, `transkript`, `kapsam_disi`) ayrılır.
2.  **RAG Tabanlı Yanıt Üretimi:** Tespit edilen niyete uygun olarak, "Bilgi Bankası"ndan (Knowledge Base) ilgili prosedür çekilir ve modelin bağlamına (context) eklenir. Böylece modelin ezbere konuşması engellenerek, sadece kurumun resmi kurallarına göre yanıt vermesi sağlanır.
3.  **Güvenlik Bariyerleri (Guardrails):** Öğrenci işleri kapsamı dışındaki sorular (fizik, genel kültür vb.) sistem tarafından tespit edilerek filtrelenir.

## Kullanılan Teknolojiler
* **Modeller:** Groq (Llama-3.3-70b-versatile) ve Google (Gemini 2.5 Flash Lite).
* **Arayüz:** Streamlit.
* **Programlama:** Python.
* **Mimari:** Intent Detection + Retrieval-Augmented Generation (Basitleştirilmiş RAG).

**Sonuç:** Geliştirilen sistem, test veri setleri üzerinde **%96'ya varan doğruluk oranıyla** niyet tespiti yapmış ve kullanıcı sorularını saniyeler içerisinde, kurum politikalarına uygun bir dille yanıtlamayı başarmıştır.

---

## 1. Model Seçimi ve Gerekçeler

Bu projede, veri setindeki niyet sınıflandırma (intent classification) görevini gerçekleştirmek için Google'ın geliştirdiği **Gemini 2.5 Flash Lite** ve Meta tarafından geliştirilen **LLaMA-3.3-70B-Versatile** modeli tercih edilmiştir.

### A) Google Gemini (Model: Gemini 2.5 Flash Lite)
* **Seçim Nedeni:** Gemini 2.5 Flash-Lite modeli, Google’ın Gemini ailesi içerisinde hız ve maliyet verimliliği odaklı bir modeldir. Özellikle küçük ve orta ölçekli veri setleriyle yapılan testlerde, hızlı yanıt süresi ve tutarlı sonuçlar üretmesi nedeniyle tercih edilmiştir.
  * **Düşük Gecikme (Low Latency):** "Lite" mimarisi sayesinde, standart modellere kıyasla işlem süresi (inference time) minimize edilmiştir. Bu özellik, binlerce satırlık öğrenci işleri   verisinin seri bir şekilde sınıflandırılması için kritik öneme sahiptir.
  * **Kaynak Verimliliği:** Model, yüksek doğruluk oranını korurken hesaplama maliyetlerini düşürmek üzere optimize edilmiştir. Özellikle kısa metinlerin (intent tespiti gibi) analizi için gereksiz parametre yükünden arındırılmış, odaklı bir performans sunar.
  * **Güncel Mimari:** Önceki serilere göre daha gelişmiş bir eğitim setine ve bağlam anlama yeteneğine sahiptir, bu da Türkçe dilindeki karmaşık öğrenci taleplerini daha iyi ayırt etmesini sağlar.

### B) Groq & Meta Llama (LLaMA-3.3-70B-Versatile)
* **Seçim Nedeni:** LLaMA-3.3-70B-Versatile modeli, Meta tarafından geliştirilen **yüksek parametreli ve açık kaynak** bir büyük dil modelidir. Modelin büyük parametre sayısı, karmaşık dil yapıları ve bağlam ilişkilerini daha iyi yakalayabilmesini sağlar.
   * **Üstün Muhakeme Yeteneği (Superior Reasoning):** 70 Milyar parametreli (70B) yapı, daha küçük modellere (örneğin 8B) kıyasla karmaşık cümle yapılarını, devrik cümleleri ve öğrencilerin dolaylı ifadelerini (bağlamı) çok daha iyi analiz eder. Bu, sınıflandırma hatalarını minimize etmek için kritik bir faktördür.
   * **Türkçe Dil Hakimiyeti:** Llama 3.3 serisi, önceki versiyonlara göre çok daha geniş birçok dilli veri setiyle eğitilmiştir. Bu durum, modelin Türkçe'deki ince nüansları ve öğrenci jargonunu (slang) ayırt etme yeteneğini önemli ölçüde artırmıştır.
   * **Karşılaştırma:** Gemini gibi kapalı bir modele karşı güçlü bir karşılaştırma imkânı sunar.

---

## 2. Kullanılan API ve Araçlar

Projenin geliştirilmesinde Python programlama dili ve aşağıdaki SDK (Software Development Kit) kütüphaneleri kullanılmıştır:

* **Google Generative AI SDK (`google-generative`):**
    * Google'ın Gemini modellerine Python üzerinden güvenli erişim sağlamak için bu resmi kütüphane kullanılmıştır.
    * Kütüphane, modelin güvenlik filtrelerini (safety settings) yapılandırmak ve deterministik (tutarlı) sonuçlar için `temperature` değerini 0'a çekmek amacıyla kullanılmıştır.
    * `gemini-2.5-flash-lite` modelinin çağrılması ve veri akışının yönetimi bu SDK üzerinden sağlanmıştır.

* **Groq Python Client (`groq`):**
    * Model ile iletişim kurmak için resmi `groq` kütüphanesi kullanılmıştır.
    * Bu kütüphane, API anahtarını otomatik olarak doğrulayarak, sunucuya gönderilen isteklerin (request) güvenli bir tünel üzerinden iletilmesini sağlar.
    * Özellikle `Llama-3.3-70b-versatile` modelinin parametrelerini yönetmek ve yanıt sürelerini optimize etmek için bu SDK tercih edilmiştir.

* **Veri İşleme Araçları:**
    * **Pandas:** CSV formatındaki veri setinin okunması ve işlenmesi.
    * **Scikit-Learn:** Modelden dönen tahminlerin doğruluğunun ölçülmesi (F1-Score, Precision, Recall hesaplamaları, Confusion matrix).

---

## 3. API Anahtarı Alımı ve Entegrasyon Bilgisi

### A) Google Gemini (Model: Gemini 2.5 Flash Lite)
Modelin projeye entegrasyonu, Google'ın bulut tabanlı AI platformu üzerinden yetkilendirme (authentication) ile gerçekleştirilmiştir.

* **Platform:** Proje için gerekli olan erişim izni, Google'ın geliştirici platformu olan **Google AI Studio** (aistudio.google.com) üzerinden sağlanmıştır.
* **Anahtar Oluşturma:** Platform üzerinde yeni bir proje tanımlanmış ve "Get API Key" modülü kullanılarak projeye özel, 2.5 Flash serisi modellere erişim yetkisi olan bir API anahtarı üretilmiştir.
* **Sisteme Entegrasyon:**
    1.  `google-generativeai` kütüphanesi projeye dahil edilmiştir (`import google.generativeai as genai`).
    2.  Üretilen API anahtarı, `genai.configure(api_key="...")` fonksiyonu ile sisteme tanıtılarak oturum açılmıştır.
    3.  Model seçimi aşamasında, hız ve performans optimizasyonu için spesifik olarak `'gemini-2.5-flash-lite'` model ismi tanımlanmış ve istemci (client) bu konfigürasyonla başlatılmıştır.

### B) Groq & Meta Llama (LLaMA-3.3-70B-Versatile)
Modelin projeye entegrasyonu, Groq Cloud platformu üzerinden yetkilendirme (authentication) ile gerçekleştirilmiştir.

* **Platform:** Proje için gerekli olan erişim izni, Groq'un geliştirici konsolu olan **Groq Cloud Console** (console.groq.com) üzerinden sağlanmıştır.
* **Anahtar Oluşturma:** Platform üzerinde "API Keys" bölümünden yeni bir anahtar oluşturulmuş ve Llama serisi modellere saniyede yüksek token işleme kapasitesiyle erişim yetkisi alınmıştır.
* **Sisteme Entegrasyon:**
    1.  `groq` kütüphanesi projeye dahil edilmiştir (`from groq import Groq`).
    2.  Üretilen API anahtarı, istemci (client) başlatılırken `client = Groq(api_key="...")` şeklinde sisteme tanıtılmıştır.
    3.  Model seçimi aşamasında, maksimum hız ve verimlilik için spesifik olarak `'llama-3.1-8b-instant'` model ismi tanımlanmış ve sorgular bu konfigürasyonla gönderilmiştir.

---

## 4. Model Performansı Karşılaştırması

İki modelin performans ölçümleri aşağıdaki gibidir:

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Groq Llama 3.3 70B** | **0.96** | **0.98** | **0.96** | **0.96** |
| Gemini 2.5 Flash Lite | 0.92 | 0.93 | 0.92 | 0.92 |

### Karşılaştırmalı Performans Analizi
Yapılan testler sonucunda elde edilen veriler ışığında, iki model arasındaki teknik farklar şu şekilde yorumlanmıştır:

**A) Genel Doğruluk (Accuracy) Farkı**
**Groq Llama 3.3 70B**, test veri setindeki örneklerin **%96'sını** doğru sınıflandırarak, rakibi Gemini 2.5 Flash Lite'a (%92) göre **%4'lük net bir başarı farkı** ortaya koymuştur. Bu fark, özellikle öğrenci niyetlerinin karmaşıklaştığı veya devrik cümlelerin kullanıldığı durumlarda, 70 milyar parametreli Llama modelinin bağlamı daha iyi kavramasından kaynaklanmaktadır.

**B) Kesinlik (Precision) Üstünlüğü**
Tabloda en dikkat çekici fark **Precision** değerinde görülmektedir. Llama 3.3, **0.98** gibi neredeyse hatasız bir kesinlik oranına ulaşmıştır. Bu durum, Llama modelinin "False Positive" (Yanlış Pozitif) oranının çok düşük olduğunu gösterir. Yani model, bir öğrenci "Transkript istiyorum" dediğinde, bunu çok yüksek bir güvenle doğru sınıfa atamakta ve başka sınıflarla (örn. Ders Kaydı) karıştırmamaktadır. Gemini (0.93) ise bu konuda daha fazla tereddüt yaşamıştır.

**C) Duyarlılık (Recall)**
**Recall** metriğinde Llama 3.3 (**0.96**), Gemini'ye (**0.92**) göre daha kapsayıcıdır. Bu, Llama modelinin veri setindeki ilgili niyetleri gözden kaçırma (miss) ihtimalinin daha düşük olduğu anlamına gelir. Özellikle "Selamlama" veya "Vedalaşma" gibi kısa ve bağlamı az olan ifadelerde Llama, Groq altyapısının hızıyla birleşen büyük model mimarisi sayesinde daha hassas bir tespit yapabilmiştir.

### D) Sonuç
Her iki model de kullanılabilecek seviyede olsa da; **Groq Llama 3.3 70B**, hem akademik ciddiyet gerektiren konulardaki yüksek kesinliği hem de genel doğruluk skoruyla projenin nihai modeli olarak belirlenmiştir. Gemini 2.5 Flash Lite ise daha düşük kaynak tüketimi gerektiren yan görevler için güçlü bir alternatif olarak konumlanabilir.
