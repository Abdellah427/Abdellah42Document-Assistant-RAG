from mistralai import Mistral

def query_mistral(user_input,history,api_key):

    # Configuration de l'API
    model = "mistral-large-latest"

    # Initialisation du client Mistral
    client = Mistral(api_key=api_key)


    # Historique des conversations (utilisation des 5 derniers messages)
    conversation_history = "\n".join(history[-5:])  # Contexte des 5 derniers messages
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
    


def load_mistral():
    api_key = "uvPKnZ4G0YFoM6KBIUkgF0KzE8dpmsgb"
    model = "mistral-embed"
    client = Mistral(api_key=api_key)
    return client, model