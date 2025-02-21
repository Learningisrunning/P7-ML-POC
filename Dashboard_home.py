import streamlit as st
import torch

torch.backends.mps.is_available = lambda: False
torch.cuda.is_available = lambda: False
import analyse_exploratoire, POC

# Création d'un menu de navigation dans la barre latérale
st.sidebar.title("Navigation")
st.sidebar.markdown("**Veuillez choisir une page pour afficher les analyses**")

page = st.sidebar.selectbox("Choisissez une page :", ["Analyse Exploratoire", "POC"])

# Amélioration de l'accessibilité avec des indicateurs textuels
if page == "Analyse Exploratoire":
    st.write("### Page : Analyse Exploratoire")
    st.write("Cette section permet d'explorer les données et d'afficher des visualisations interactives.")
    analyse_exploratoire.show()
    
elif page == "POC":
    st.write("### Page : POC (Proof of Concept)")
    st.write("Cette section permet de tester un modèle de prédiction basé sur les données sélectionnées.")
    POC.show()
