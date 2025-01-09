from mistralai import Mistral
import streamlit as st
import os

# Configuration de l'API
os.environ["MISTRAL_API_KEY"] = "ZqdIf3aZDPfbyWXCuHy2WhDmdYCTZ935"

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

# Initialisation du client Mistral
client = Mistral(api_key=api_key)

def mistral_response(user_input):
    if 'history' not in st.session_state:
    st.session_state.history = []
    # Historique des conversations (utilisation des 5 derniers messages)
    conversation_history = "\n".join(st.session_state.history[-5:])  # Contexte des 5 derniers messages
    prompt = f"{conversation_history}\nYou: {user_input}\nBot:"
    
    try:
        # Envoi de la requête au modèle
        response = client.chat.complete(   
            model=model,  
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=150,  # Limite de longueur des réponses
            temperature=0.7,  # Contrôle de la créativité
            top_p=0.9,  # Filtrage par probabilité cumulative
        )
        return response.choices[0].message.content

    except Exception as e:
        # Gestion des erreurs
        print(f"Erreur détaillée : {e}")
        return "Désolé, une erreur est survenue lors du traitement de votre demande."

# Fonction principale de l'application Streamlit
def main():
    # Initialiser l'historique
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Titre de l'application
    st.title("Chatbot avec l'API Mistral")

    # Formulaire pour l'entrée utilisateur
    form = st.form(key='chat_form')
    user_input = form.text_input("You:", "")
    submit_button = form.form_submit_button(label='Send')

    # Gestion de la soumission du formulaire
    if submit_button and user_input:
        response = mistral_response(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

    # Affichage de l'historique des conversations
    for message in st.session_state.history:
        st.write(message)

if __name__ == "__main__":
    main()
