import streamlit as st
import openai
import faiss
import numpy as np

# 🔐 OpenAI API-Key
openai.api_key = "sk-..."  # Deinen echten Key hier einfügen

# 👉 Hier deine gespeicherten Vektoren/Textblöcke einfügen
# Diese müssen entweder direkt im Code stehen oder über Pickle/Festplatte geladen werden
# Beispiel (Platzhalter):
# texts = [...]
# vectors = [...]
# → oder z. B. aus .pkl laden

# Dummy-Daten fürs Beispiel
texts = ["§ 2 UrlG\nJeder Arbeitnehmer hat Anspruch auf 5 Wochen Urlaub.",
         "§ 3 UrlG\nNach 25 Dienstjahren sind es 6 Wochen Urlaub."]
vectors = [openai.embeddings.create(input=t, model="text-embedding-3-small").data[0].embedding for t in texts]

index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

# UI
st.title("🧑‍⚖️ Arbeitsrechts-Agent Österreich")
frage = st.text_input("Stell deine arbeitsrechtliche Frage:")

if st.button("Frage stellen") and frage:
    frage_embed = openai.embeddings.create(
        input=frage,
        model="text-embedding-3-small"
    ).data[0].embedding

    D, I = index.search(np.array([frage_embed]).astype("float32"), 3)
    relevante_texte = [texts[i] for i in I[0]]

    # GPT-Prompt
    prompt = f"""
Du bist ein juristischer KI-Assistent für österreichisches Arbeitsrecht.

Hier sind relevante Gesetzesauszüge:
\n\n{relevante_texte[0]}\n\n
{relevante_texte[1] if len(relevante_texte) > 1 else ''}
{relevante_texte[2] if len(relevante_texte) > 2 else ''}

Bitte beantworte folgende Frage:
„{frage}“

🔹 Verwende ausschließlich die sichtbaren Gesetzesquellen.
🔹 Gib Paragraphen korrekt an (z. B. § 2 UrlG).
🔹 Wenn unklar: „nicht eindeutig geregelt“.
"""

    antwort = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    st.markdown("**🤖 Antwort:**")
    st.write(antwort)
