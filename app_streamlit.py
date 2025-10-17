import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# --- 1. IMPORTS NÉCESSAIRES POUR VOTRE MODÈLE ---
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Chargement du modèle yfinance (inchangé) ---
try:
    import yfinance as yf
except Exception:
    yf = None

# --- Mise en cache pour accélérer le chargement ---
@st.cache_data
def load_all_market_data(folder: str) -> pd.DataFrame:
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {pattern}")
    dataframes = [pd.read_csv(fp) for fp in files]
    df = pd.concat(dataframes, ignore_index=True)
    return df

# --- Mise en cache pour le modèle LSTM ---
@st.cache_resource
def load_lstm_model(model_path: str):
    """Charge le modèle Keras une seule fois."""
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

# --- FONCTION POUR UTILISER LE MODÈLE LSTM ---
def forecast_future_with_lstm(df: pd.DataFrame, horizon_days: int = 60, model_path: str = "lstm_market_predictor.h5") -> pd.DataFrame | None:
    """Prédit les futurs cours en utilisant le modèle LSTM pré-entraîné."""
    model = load_lstm_model(model_path)
    if model is None:
        st.warning(f"Le modèle {model_path} est introuvable. La prédiction LSTM est désactivée.")
        return None

    sequence_length = 60 # Doit correspondre à la longueur de séquence de l'entraînement !
    
    # 1. Préparer les données
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # 2. Prendre la dernière séquence de données connues
    last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
    
    # 3. Prédire pas à pas pour l'horizon demandé
    predicted_prices_scaled = []
    current_sequence = last_sequence
    
    for _ in range(horizon_days):
        next_pred_scaled = model.predict(current_sequence)[0, 0]
        predicted_prices_scaled.append(next_pred_scaled)
        # Mettre à jour la séquence en ajoutant la nouvelle prédiction et en retirant la plus ancienne valeur
        new_sequence_entry = np.array([[next_pred_scaled]]).reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], new_sequence_entry, axis=1)

    # 4. Inverser la normalisation pour obtenir les vrais prix
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    
    # 5. Créer le DataFrame avec les dates futures
    import pandas.tseries.offsets as offsets
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + offsets.BDay(1), periods=horizon_days, freq='B')
    
    return pd.DataFrame({"PredictedClose": predicted_prices.flatten()}, index=future_dates)


def prepare_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    cols_needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    sub = df[df["Ticker"] == ticker][cols_needed].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub.sort_values("Date", inplace=True)
    sub.set_index("Date", inplace=True)
    return sub

def plot_prices_with_actions(df: pd.DataFrame, actions: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"))
    if actions is not None and not actions.empty:
        buy_idx = actions[actions == 1].index
        sell_idx = actions[actions == 2].index
        fig.add_trace(go.Scatter(
            x=buy_idx, y=df.loc[buy_idx, "Close"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
        ))
        fig.add_trace(go.Scatter(
            x=sell_idx, y=df.loc[sell_idx, "Close"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", color="red", size=10),
        ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=500)
    return fig

def sma_crossover_actions(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    closes = df["Close"].astype(float)
    sma_fast = closes.rolling(fast, min_periods=1).mean()
    sma_slow = closes.rolling(slow, min_periods=1).mean()
    signal = (sma_fast > sma_slow).astype(int)
    action = pd.Series(0, index=df.index)
    prev = signal.shift(1).fillna(0).astype(int)
    action[(prev == 0) & (signal == 1)] = 1
    action[(prev == 1) & (signal == 0)] = 2
    return action

def simulate_portfolio(df: pd.DataFrame, actions: pd.Series, initial_balance: float = 10_000.0) -> pd.DataFrame:
    balance = initial_balance
    shares_held = 0.0
    net_worth_list: list[float] = []
    # Correction pour gérer les dates manquantes dans les actions
    for idx in df.index:
        action = actions.get(idx, 0) # Utiliser .get() pour éviter les erreurs
        price = float(df.at[idx, "Close"])
        if action == 1 and balance > price:
            shares_held += 1.0
            balance -= price
        elif action == 2 and shares_held > 0.0:
            shares_held -= 1.0
            balance += price
        net_worth_list.append(balance + shares_held * price)
    result = pd.DataFrame(
        {"NetWorth": net_worth_list},
        index=df.index,
    )
    return result

def fetch_recent_market_data_yf(ticker: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
    if yf is None: return None
    try:
        dfy = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
        if dfy is None or dfy.empty: return None
        dfy = dfy.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        dfy = dfy[["Open", "High", "Low", "Close", "Volume"]]
        dfy.index = pd.to_datetime(dfy.index)
        return dfy
    except Exception:
        return None

def forecast_future_with_ets_seasonal(df: pd.DataFrame, horizon_days: int = 60) -> pd.DataFrame | None:
    closes = df["Close"].astype(float)
    if len(closes) < 252 * 2:
        return None 
    try:
        model = ExponentialSmoothing(
            closes, trend="add", seasonal="add", seasonal_periods=252, initialization_method="estimated"
        )
        fit = model.fit()
        preds = fit.forecast(horizon_days)
    except Exception as e:
        print("Erreur ETS seasonal:", e)
        return None
    import pandas.tseries.offsets as offsets
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + offsets.BDay(1), periods=horizon_days, freq='B')
    return pd.DataFrame({"PredictedClose": preds.values}, index=future_dates)

def main() -> None:
    st.set_page_config(page_title="IA Bourse Invest", layout="wide")
    st.title("IA Bourse Invest — Visualisation et Bot de Trading")
    data_folder = "Global Stock Market (2008-2023)"
    
    df_all = load_all_market_data(data_folder)
    
    tickers = sorted(df_all["Ticker"].dropna().unique())
    with st.sidebar:
        st.header("Paramètres")
        ticker = st.selectbox("Sélectionnez une action (Ticker)", options=tickers, index=0 if tickers else None)
        date_min = pd.to_datetime(df_all["Date"]).min() if len(df_all) else datetime(2010, 1, 1)
        date_max = pd.to_datetime(df_all["Date"]).max() if len(df_all) else datetime(2024, 1, 1)
        date_range = st.date_input("Plage de dates", value=(date_min, date_max))
        run_button = st.button("Lancer la simulation")
    if not ticker:
        st.info("Sélectionnez un ticker pour commencer.")
        return
    df_ticker = prepare_ticker_df(df_all, ticker)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        df_ticker = df_ticker.loc[(df_ticker.index >= start_date) & (df_ticker.index <= end_date)]
    st.subheader(f"Cours — {ticker}")
    base_fig = plot_prices_with_actions(df_ticker)
    actions_series: pd.Series | None = None
    with st.sidebar:
        st.markdown("---")
        extend_real = st.toggle("Étendre avec données réelles (Yahoo)", value=True)
        show_forecast = st.toggle("Afficher prédictions (futures)", value=True)
        # --- 3. SÉLECTEUR DE MODÈLE ---
        model_choice = st.selectbox("Choisissez le modèle de prédiction", ["ETS (Statistique)", "LSTM (IA)"])
        auto_predict_years = st.slider(
            "Durée de prédiction (années futures)", min_value=1, max_value=5, value=2, step=1
        )

    df_for_forecast = df_ticker
    if extend_real and yf is not None:
        last_date = df_ticker.index.max()
        start_real = last_date + pd.Timedelta(days=1)
        recent_real = fetch_recent_market_data_yf(ticker, start_real)
        if recent_real is not None and not recent_real.empty:
            base_fig.add_trace(go.Scatter(x=recent_real.index, y=recent_real["Close"], name="Données réelles récentes", mode="lines", line=dict(color="#1f77b4")))
            df_for_forecast = pd.concat([df_ticker, recent_real])

    if show_forecast:
        horizon_days = 252 * auto_predict_years
        fut = None
        # --- 4. APPEL DE LA BONNE FONCTION DE PRÉDICTION ---
        if model_choice == "ETS (Statistique)":
            fut = forecast_future_with_ets_seasonal(df_for_forecast, horizon_days=horizon_days)
            pred_name = "Prévision ETS"
        else: # LSTM
            with st.spinner("Prédiction LSTM en cours..."):
                fut = forecast_future_with_lstm(df_for_forecast, horizon_days=horizon_days)
            pred_name = "Prévision LSTM"
            
        if fut is not None and not fut.empty:
            base_fig.add_trace(go.Scatter(
                x=fut.index, y=fut["PredictedClose"], name=pred_name, mode="lines",
                line=dict(color="#2aff0e", dash="dash"),
            ))
            
    st.plotly_chart(base_fig, use_container_width=True)
    
    if run_button:
        with st.spinner("Simulation en cours..."):
            actions_series = sma_crossover_actions(df_ticker)
        result = simulate_portfolio(df_ticker, actions_series)
        st.subheader("Résultat de la simulation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valeur finale (Net Worth)", f"{result['NetWorth'].iloc[-1]:,.2f} €")
        with col2:
            roi = (result["NetWorth"].iloc[-1] - 10_000.0) / 10_000.0 * 100.0
            st.metric("ROI", f"{roi:.2f}%")
        st.plotly_chart(plot_prices_with_actions(df_ticker, actions_series), use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=result.index, y=result["NetWorth"], name="Net Worth", mode="lines"))
        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400)
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()

