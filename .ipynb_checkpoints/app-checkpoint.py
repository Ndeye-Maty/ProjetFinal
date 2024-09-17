import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Prédiction des Prix de l'Immobilier", layout="wide")

# Charger les données
df = pd.read_csv("USA_Housing.csv")

# Charger l'image
image = Image.open("63cfb4827afa0-AdobeStock_5597188821.jpg")

# Barre latérale pour la navigation
st.sidebar.title("Options")
page = st.sidebar.selectbox("Choisissez une option", ["Accueil", "Exploration des données", "Prédiction"])

# Prétraitement des données
df.drop(columns=['Address'], inplace=True)  # Supprimer la colonne 'Address'
X = df[["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Area Population"]]
y = df["Price"]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Page d'accueil avec l'image
if page == "Accueil":
  
    # Texte grand et stylisé
    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>Bienvenue dans l'application de prédiction des prix de l'immobilier</h1>
        <h2 style='text-align: center;'>Cette application vous permet de prédire les prix de l'immobilier en fonction de plusieurs caractéristiques.</h2>
    """, unsafe_allow_html=True)
    
    # Afficher l'image
    width = 800
    height = int((width / image.width) * image.height)
    st.image(image, caption="Bienvenue dans l'application de prédiction des prix de l'immobilier", width=width)

# Page d'exploration des données
elif page == "Exploration des données":
    st.header("Exploration des données")
    # Afficher les premières lignes du DataFrame
    st.subheader("Aperçu des données")
    st.write(df)
    
    # Afficher des graphiques
    st.subheader("Distribution des prix")
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Corrélation des caractéristiques")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Page de prédiction
elif page == "Prédiction":
    st.header("Prédiction du Prix de l'Immobilier")
    
    # Utilisation du modèle de Régression Linéaire
    model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Affichage du MAE
    st.write(f"**Mean Absolute Error du modèle :** ${mae:,.2f}")
    
    # Entrée des caractéristiques par l'utilisateur
    st.sidebar.subheader("Entrer les caractéristiques")
    revenue = st.sidebar.slider("Revenu moyen de la zone", min_value=10000, max_value=200000, value=50000)
    house_age = st.sidebar.slider("Âge moyen des maisons", min_value=0.0, max_value=100.0, value=15.0)
    rooms = st.sidebar.slider("Nombre moyen de pièces", min_value=1.0, max_value=10.0, value=6.0)
    bedrooms = st.sidebar.slider("Nombre moyen de chambres", min_value=1.0, max_value=10.0, value=3.0)
    population = st.sidebar.slider("Population de la zone", min_value=0, max_value=100000, value=20000)
    
    # Préparer les données d'entrée pour la prédiction
    input_data = np.array([[revenue, house_age, rooms, bedrooms, population]])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    
    # Afficher le résultat
    st.write(f"### Le prix prédit est de **${predicted_price:,.2f}** (en USD)")
    
    # Graphique de comparaison avec les données réelles
    st.subheader("Comparaison avec les données réelles")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax.set_xlabel('Prix Réels')
    ax.set_ylabel('Prix Prédits')
    st.pyplot(fig)
