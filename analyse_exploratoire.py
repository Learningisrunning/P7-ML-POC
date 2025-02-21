import streamlit as st
import pandas as pd
import plotly.express as px 

def show():
    file_path = "Data.csv"
    try:
        df = pd.read_csv(file_path)
        st.write("### Aperçu des données")
        st.write(df.describe())

        # Histogramme avec amélioration de l'accessibilité
        st.write("### Histogramme du nombre de visiteurs")
        fig = px.histogram(
            df, x="total_guests", nbins=30, histnorm='density', opacity=0.6,
            title="Répartition des visiteurs",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig)

        with st.expander("Voir les données utilisées pour l'histogramme"):
            st.write(df[['total_guests']])

        # Graphique circulaire
        df_grouped = df.groupby("lounge_name").sum()
        st.write("### Répartition des visiteurs par lounge")
        fig_pie = px.pie(
            df_grouped, values="total_guests", names=df_grouped.index,
            title="Proportion des visiteurs par lounge",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_pie)

        with st.expander("Voir les données utilisées pour le graphique circulaire"):
            st.write(df_grouped[['total_guests']])
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
