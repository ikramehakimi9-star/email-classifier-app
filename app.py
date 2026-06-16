import string
import pickle
import streamlit as st
import streamlit_analytics
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Téléchargement des dépendances NLTK obligatoires
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# 2. Initialisation des outils de texte
ps = PorterStemmer()
stop_words_set = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

def transform_text(text):
    # Passage en minuscules
    text = text.lower()
    
    # Tokenisation
    tokens = nltk.word_tokenize(text)
    
    # Filtrage : Garder uniquement l'alphanumérique, retirer stopwords et ponctuations
    clean_tokens = [
        ps.stem(word) for word in tokens 
        if word.isalnum() and word not in stop_words_set and word not in punctuation_set
    ]
    
    # Reconstitution du texte
    return " ".join(clean_tokens)

# 3. Chargement des modèles sauvegardés
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Erreur de chargement : Assurez-vous que les fichiers 'vectorizer.pkl' et 'model.pkl' sont dans le même dossier que ce script. ({e})")
    st.stop()

# 4. Interface Streamlit
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():  # .strip() évite de valider si l'utilisateur met juste des espaces
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]  # Récupération de la valeur (0 ou 1)
        
        # 4. Display
        if result == 1:
            st.header("Spam 🚨")
        else:
            st.header("Not Spam ✅")
    else:
        st.warning("Please enter a valid message before predicting.")
