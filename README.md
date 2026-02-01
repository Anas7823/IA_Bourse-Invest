# IA Bourse Invest

Un projet d'intelligence artificielle pour la prédiction des marchés boursiers et le trading automatisé utilisant des modèles avancés comme LSTM, Temporal Fusion Transformer (TFT) et Deep Q-Network (DQN).

## Description

Ce projet explore l'utilisation de l'IA pour analyser et prédire les cours des actions. Il comprend :

- **Prétraitement des données** : Chargement et nettoyage des données boursières historiques.
- **Modèles de prédiction** : Utilisation de LSTM pour les prédictions temporelles et TFT pour des prévisions plus sophistiquées.
- **Trading automatisé** : Agent DQN pour prendre des décisions de trading en apprentissage par renforcement.
- **Visualisation** : Applications Streamlit pour explorer les données, simuler des stratégies et afficher des prédictions.

Le projet est basé sur des données historiques de marchés mondiaux (2008-2023) et peut être étendu avec des données en temps réel via Yahoo Finance.

## Fonctionnalités

- Chargement et enrichissement des données avec des indicateurs techniques (SMA, RSI, Bandes de Bollinger).
- Entraînement de modèles LSTM pour prédire les prix futurs.
- Intégration de TFT pour des prévisions multi-séries temporelles.
- Environnements de trading personnalisés pour l'apprentissage par renforcement avec DQN.
- Simulations de portefeuilles avec stratégies simples (croisement de moyennes mobiles).
- Applications Streamlit pour l'interaction utilisateur : visualisation des cours, prédictions futures et backtesting.

## Installation

### Prérequis

- Python 3.8+
- Bibliothèques : pandas, numpy, tensorflow, pytorch, pytorch-forecasting, stable-baselines3, streamlit, yfinance, statsmodels, scikit-learn, gymnasium, etc.

### Étapes

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/Anas7823/IA_Bourse-Invest.git
   cd IA_Bourse-Invest

2. Créez un environnement virtuel :
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt

4. Configurez les variables d'environnement dans .env (ex : YF_TICKERS = ^VIX).

## Utilisation

Notebooks Jupyter
Bourse.ipynb : Prétraitement, entraînement LSTM et DQN basique.
Bourse_TFT.ipynb : Intégration TFT et DQN multi-actions.
Bourse_v2.ipynb : Versions alternatives ou expérimentales.

Lancez avec :
    ```bash
    jupyter notebook

Applications Streamlit
app_streamlit.py : Interface pour visualisation et simulation avec données locales.
app_streamlit_api.py : Version utilisant Yahoo Finance pour des données en temps réel.

Lancez avec :
    ```bash
    streamlit run app_streamlit.py

Modèles pré-entraînés

lstm_market_predictor.h5 et lstm_stock_predictor.h5 : Modèles LSTM sauvegardés.
Modèles DQN : Sauvegardés dans des dossiers comme dqn_tft_trader.


## Structure du Projet
.
├── .env                          # Variables d'environnement
├── .gitignore                    # Fichiers ignorés par Git
├── app_streamlit.py              # App Streamlit principale
├── app_streamlit_api.py          # App Streamlit avec API Yahoo
├── Bourse.ipynb                  # Notebook LSTM + DQN
├── Bourse_TFT.ipynb              # Notebook TFT + DQN
├── Bourse_v2.ipynb               # Notebook alternatif
├── lstm_market_predictor.h5      # Modèle LSTM 1
├── lstm_stock_predictor.h5       # Modèle LSTM 2
├── dqn_multistock_tensorboard/   # Logs TensorBoard DQN multi-stock
├── dqn_trading_tensorboard/      # Logs TensorBoard DQN trading
└── Global Stock Market (2008-2023)/  # Données CSV
    ├── 2008_Global_Markets_Data.csv
    └── ...