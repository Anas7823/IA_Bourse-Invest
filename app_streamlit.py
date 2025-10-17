import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import requests
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# On essaie de récupérer les variables
api_key = os.getenv("FMP_API_KEY")
tickers = os.getenv("FMP_TICKERS")

try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_market_data_from_fmp(
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Récupère les historiques OHLCV depuis Financial Modeling Prep.
    Exige api_key (ou variable d'env FMP_API_KEY) et une liste de tickers.
    Retourne un DataFrame avec colonnes ['Date','Open','High','Low','Close','Volume','Ticker'].
    """
    api_key = api_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY manquant pour fetch_market_data_from_fmp")

    rows: list[dict] = []
    session = requests.Session()
    for ticker in tickers:
        url = (
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
            f"?apikey={api_key}"
        )
        if start:
            url += f"&from={start}"
        if end:
            url += f"&to={end}"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            historical = payload.get("historical") or []
            for item in historical:
                rows.append({
                    "Ticker": ticker,
                    "Date": item.get("date"),
                    "Open": item.get("open"),
                    "High": item.get("high"),
                    "Low": item.get("low"),
                    "Close": item.get("close"),
                    "Volume": item.get("volume"),
                })
        except Exception as exc:
            print(f"Erreur FMP pour {ticker}: {exc}")
            continue

    if not rows:
        raise RuntimeError("Aucune donnée récupérée depuis FMP")

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df.reset_index(drop=True)


def load_all_market_data() -> pd.DataFrame:
    """
    Charge les données du marché exclusivement depuis l'API Financial Modeling Prep (FMP).
    Exige que les variables d'environnement FMP_API_KEY et FMP_TICKERS soient définies.

    - FMP_TICKERS: 'AAPL,MSFT,^GSPC' (liste de tickers séparés par des virgules)
    - Optionnel: FMP_FROM, FMP_TO (YYYY-MM-DD)

    Lève une ValueError si les variables requises ne sont pas trouvées.
    """
    api_key = os.getenv("FMP_API_KEY")
    tickers_env = os.getenv("FMP_TICKERS")

    # Vérification que les variables d'environnement sont bien présentes
    if not api_key or not tickers_env:
        raise ValueError(
            "Configuration API manquante. "
            "Veuillez définir les variables d'environnement FMP_API_KEY et FMP_TICKERS."
        )

    # Logique de l'API (conservée de l'original)
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
    start = os.getenv("FMP_FROM")  # facultatif
    end = os.getenv("FMP_TO")      # facultatif
    
    print(f"🔌 Chargement via l'API FMP pour {len(tickers)} tickers")
    return fetch_market_data_from_fmp(tickers=tickers, start=start, end=end, api_key=api_key)

def prepare_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Prépare le DataFrame pour un ticker donné.
    Fonction inchangée mais robuste au DataFrame FMP (qui a déjà 'Ticker').
    """
    cols_needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if "Ticker" in df.columns:
        sub = df[df["Ticker"] == ticker][cols_needed].copy()
    else:
        sub = df[df["Ticker"] == ticker][cols_needed].copy() if "Ticker" in df.columns else df[cols_needed].copy()
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
    for idx, action in actions.items():
        price = float(df.at[idx, "Close"])
        if action == 1 and balance > price:
            shares_held += 1.0
            balance -= price
        elif action == 2 and shares_held > 0.0:
            shares_held -= 1.0
            balance += price
        net_worth_list.append(balance + shares_held * price)
    result = pd.DataFrame(
        {
            "Close": df["Close"].values,
            "Action": actions.values,
            "NetWorth": net_worth_list,
        },
        index=df.index,
    )
    return result

def fetch_recent_market_data_yf(ticker: str, start_date: pd.Timestamp) -> pd.DataFrame | None:
    if yf is None:
        return None
    try:
        dfy = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=False)
        if dfy is None or dfy.empty:
            return None
        dfy = dfy.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
        dfy.index = pd.to_datetime(dfy.index)
        return dfy
    except Exception:
        return None

def forecast_future_with_ets_seasonal(df: pd.DataFrame, horizon_days: int = 60) -> pd.DataFrame | None:
    closes = df["Close"].astype(float)
    if len(closes) < 252 * 2:
        return None  # Besoin de deux ans mini pour saisonnalité annuelle
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
    future_dates = []
    cur = last_date
    while len(future_dates) < horizon_days:
        cur = cur + offsets.BDay(1)
        future_dates.append(cur)
    return pd.DataFrame({"PredictedClose": preds.values}, index=pd.DatetimeIndex(future_dates))

def main() -> None:
    st.set_page_config(page_title="IA Bourse Invest", layout="wide")
    st.title("IA Bourse Invest — Visualisation et Bot de Trading")
    with st.spinner("Chargement des données..."):
        try:
            df_all = load_all_market_data()
        except Exception as exc:
            st.error(f"Erreur de chargement : {exc}")
            return
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
    st.sidebar.markdown("---")
    extend_real = st.sidebar.toggle("Étendre avec données réelles (Yahoo)", value=True)
    show_forecast = st.sidebar.toggle("Afficher prédictions (futures)", value=True)
    auto_predict_years = st.sidebar.slider(
        "Durée de prédiction (années futures)", min_value=1, max_value=5, value=2, step=1
    )
    df_for_forecast = df_ticker
    if extend_real:
        last_date = df_ticker.index.max()
        try:
            import pandas.tseries.offsets as offsets
            start_real = last_date + offsets.BDay(1)
        except Exception:
            start_real = last_date + pd.Timedelta(days=1)
        recent_real = fetch_recent_market_data_yf(ticker, start_real)
        if recent_real is not None and not recent_real.empty:
            base_fig.add_trace(
                go.Scatter(
                    x=recent_real.index,
                    y=recent_real["Close"],
                    name="Données réelles récentes",
                    mode="lines",
                    line=dict(color="#1f77b4"),
                )
            )
            df_for_forecast = pd.concat([df_ticker, recent_real])
    if show_forecast:
        horizon_days = 252 * auto_predict_years
        fut = forecast_future_with_ets_seasonal(
            df_for_forecast,
            horizon_days=horizon_days
        )
        if fut is not None and not fut.empty:
            base_fig.add_trace(
                go.Scatter(
                    x=fut.index,
                    y=fut["PredictedClose"],
                    name=f"Prévision ETS saisonnière ({horizon_days} jours ouvrés)",
                    mode="lines",
                    line=dict(color="#ff7f0e", dash="dash"),
                )
            )
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
