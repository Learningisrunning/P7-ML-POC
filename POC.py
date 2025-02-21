import streamlit as st
import pandas as pd
import plotly.express as px 
from fonction import transforme_data_selected_for_TFT, prediction, custom_to_prediction_dataframe

import nest_asyncio

def show():
    nest_asyncio.apply()
    file_path = "test_data.csv"
    try:
        df = pd.read_csv(file_path)
        
        st.write("### Sélection des paramètres")
        liste_annees = df["year"].unique()
        liste_salons = df["lounge_name"].unique()

        annee_selectionnee = st.selectbox("Sélectionnez l'année", liste_annees)
        salon_selectionne = st.selectbox("Sélectionnez le salon", liste_salons)
        st.write(f"Vous avez sélectionné : **{annee_selectionnee}** et **{salon_selectionne}**")
        
        if st.button("Lancer la prédiction"):
            df_filtered = df[(df["lounge_name"] == salon_selectionne) & (df["year"] == annee_selectionnee)]
            df_filtered.drop("year", axis=1, inplace=True)
            
            st.write("### Données filtrées utilisées pour la prédiction")
            with st.expander("Voir les données brutes utilisées"):
                st.write(df_filtered)
            
            prediction_dataloader = transforme_data_selected_for_TFT(df_filtered)
            raw_predictions, x = prediction(prediction_dataloader)
            predictions = custom_to_prediction_dataframe(x, raw_predictions)
            
            st.write("### Résultats de la prédiction")
            with st.expander("Voir les résultats en détail"):
                st.write(predictions)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
