import streamlit as st
from groq import Groq

st.set_page_config(page_title="İTÜ Asistanı", page_icon="🎓", layout="wide")

st.sidebar.title("⚙️ Ayarlar")
groq_api_key = st.sidebar.text_input("Groq API Key (gsk_...)", type="password")
if st.sidebar.button("Sohbeti Sıfırla"):
    st.session_state.messages = []
    st.rerun()

intent_responses = {
    "ders_kaydi": "Ders kayıtları 20-24 Eylül tarihleri arasında SIS üzerinden yapılacaktır. 1. sınıfların kaydı otomatiktir.",
    "transkript": "Resmi transkript belgenizi E-Devlet üzerinden veya Maslak Kampüsü Öğrenci İşleri Daire Başkanlığı'ndan alabilirsiniz.",
    "yatay_gecis": "Kurum içi yatay geçiş için AGNO en az 2.50, kurumlar arası geçiş için ise en az 3.00 olmalıdır.",
    "selamlama": "Merhaba! Ben İTÜ Öğrenci İşleri Asistanıyım. Sadece ders kaydı, transkript ve yatay geçiş konularında yardımcı olabilirim.",
    "vedalasma": "İyi günler dilerim, eğitim hayatınızda başarılar!",
    "kapsam_disi": "Üzgünüm, sadece öğrenci işleri konularında yardımcı olabilirim. Diğer konularda cevap verecek yetkinliğe sahip değilim."
}

def detect_intent(user_input, client):
    system_prompt = """
    Sen bir sınıflandırma modelisin. Görevin gelen mesajın konusunu tespit etmek.
    
    Kategoriler:
    1. ders_kaydi (Ders seçimi, SIS, kayıt tarihleri vb.)
    2. transkript (Not dökümü, belge alma vb.)
    3. yatay_gecis (Ortalama, geçiş şartları vb.)
    4. selamlama (Merhaba, selam vb.)
    5. vedalasma (Görüşürüz, bay bay vb.)
    6. kapsam_disi (BUNLARIN DIŞINDAKI HER ŞEY. Örn: Hava durumu, kod yazma, başkentler, futbol vb.)

    Sadece kategori ismini yaz. Başka hiçbir şey yazma.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=10
        )
        return completion.choices[0].message.content.strip()
    except:
        return "kapsam_disi"

def generate_answer(user_input, intent, client):
    if intent == "kapsam_disi":
        return intent_responses["kapsam_disi"]

    context_info = intent_responses.get(intent, "")
    
    system_prompt = f"""
    Sen yardımsever bir üniversite asistanısın.
    
    Kural 1: Sadece sana verilen şu bilgiyi kullanarak cevap ver: "{context_info}"
    Kural 2: Bu bilgi dışına çıkma, uydurma yapma.
    Kural 3: Kullanıcıya nazik ol.
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3,
        max_tokens=250
    )
    return completion.choices[0].message.content

st.title("🎓 İTÜ Öğrenci İşleri Botu")
st.info("Bu bot ile ders kaydı, transkript ve yatay geçiş konularında yardım alabilirsiniz.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    if not groq_api_key:
        st.error("Lütfen API anahtarını girin.")
        st.stop()

    client = Groq(api_key=groq_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        detected_intent = detect_intent(prompt, client)
        
        if detected_intent == "kapsam_disi":
            full_response = intent_responses["kapsam_disi"]
        else:
            full_response = generate_answer(prompt, detected_intent, client)
        
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
