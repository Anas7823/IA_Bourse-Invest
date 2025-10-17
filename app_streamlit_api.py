import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dotenv import load_dotenv
try:
    import yfinance as yf
except ImportError:
    st.error("La bibliothèque 'yfinance' n'est pas installée. Veuillez l'installer en exécutant : pip install yfinance")
    # Arrête l'exécution du script si la bibliothèque est essentielle
    st.stop()
 
load_dotenv()
# Configuration proxy (réseau d'entreprise) via .env
# Variables supportées: PROXY, HTTP_PROXY, HTTPS_PROXY, NO_PROXY
_proxy = os.getenv("PROXY") or os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
if _proxy:
	os.environ["HTTP_PROXY"] = _proxy
	os.environ["HTTPS_PROXY"] = _proxy
_no_proxy = os.getenv("NO_PROXY")
if _no_proxy:
	os.environ["NO_PROXY"] = _no_proxy

try:
	import yfinance as yf
except Exception:
	yf = None


def fetch_market_data_yf(
	tickers: list[str],
	start: str = None,
	end: str = None,
) -> pd.DataFrame:
	rows = []
	for ticker in tickers:
		print(f"Yahoo Finance download for: {ticker}")
		df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
		if df is None or df.empty:
			continue
		for idx, row in df.iterrows():
			rows.append({
				"Ticker": ticker,
				"Date": idx,
				"Open": row["Open"],
				"High": row["High"],
				"Low": row["Low"],
				"Close": row["Close"],
				"Volume": row["Volume"],
			})
	if not rows:
		raise RuntimeError("Aucune donnée récupérée depuis Yahoo Finance")
	df = pd.DataFrame(rows)
	df["Date"] = pd.to_datetime(df["Date"])
	df.sort_values(["Ticker", "Date"], inplace=True)
	return df.reset_index(drop=True)


def load_all_market_data() -> pd.DataFrame:
	tickers_env = os.getenv("YF_TICKERS")
	if not tickers_env:
		raise ValueError("Veuillez définir YF_TICKERS dans votre fichier .env")
	tickers = [t.strip() for t in tickers_env.split(",") if t.strip()]
	start = os.getenv("YF_FROM")  # optionnel
	end = os.getenv("YF_TO")      # optionnel
	print(f"🔌 Chargement Yahoo Finance pour {len(tickers)} tickers")
	return fetch_market_data_yf(tickers, start=start, end=end)


def prepare_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
	cols_needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
	sub = df[df["Ticker"] == ticker][cols_needed].copy()
	sub["Date"] = pd.to_datetime(sub["Date"])
	sub.sort_values("Date", inplace=True)
	sub.set_index("Date", inplace=True)
	return sub


def map_to_yahoo_symbol(dataset_ticker: str) -> str:
	"""Convertit quelques alias courants vers le symbole Yahoo attendu."""
	t = str(dataset_ticker).strip().upper()
	aliases = {
		"GSPC": "^GSPC",
		"SPX": "^GSPC",
		"S&P500": "^GSPC",
		"S&P 500": "^GSPC",
		"DJI": "^DJI",
		"NDX": "^NDX",
		"FCHI": "^FCHI",
	}
	return aliases.get(t, dataset_ticker)


def plot_prices_with_actions(df: pd.DataFrame, actions: pd.Series | None = None) -> alt.Chart:
	base = alt.Chart(df.reset_index()).mark_line(color="#1f77b4").encode(
		x=alt.X("Date:T", title="Date"),
		y=alt.Y("Close:Q", title="Close"),
		tooltip=["Date:T", "Close:Q"],
	)
	layers = [base]
	if actions is not None and not actions.empty:
		actions_df = pd.DataFrame({
			"Date": actions.index,
			"Action": actions.values,
			"Close": df.loc[actions.index, "Close"].values,
		}).reset_index(drop=True)
		buy = actions_df[actions_df["Action"] == 1]
		sell = actions_df[actions_df["Action"] == 2]
		if not buy.empty:
			layers.append(
				alt.Chart(buy).mark_point(color="green", shape="triangle-up", size=80).encode(
					x="Date:T", y="Close:Q", tooltip=["Date:T", "Close:Q"]
				)
			)
		if not sell.empty:
			layers.append(
				alt.Chart(sell).mark_point(color="red", shape="triangle-down", size=80).encode(
					x="Date:T", y="Close:Q", tooltip=["Date:T", "Close:Q"]
				)
			)
	return alt.layer(*layers).properties(height=500)


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
	st.title("IA Bourse Invest — Visualisation et Bot de Trading (Yahoo Finance Data)")
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
  	# 1. Forcer la colonne 'Close' à être numérique. Les erreurs deviendront NaN.
	df_ticker['Close'] = pd.to_numeric(df_ticker['Close'], errors='coerce')
	# 2. (Recommandé) Supprimer les lignes où la conversion a échoué.
	df_ticker.dropna(subset=['Close'], inplace=True)
	
 	if df_ticker.empty:
		st.warning("Aucune donnée disponible pour le ticker et la plage de dates sélectionnés. Veuillez ajuster vos paramètres.")
		return  # Arrête l'exécution pour éviter l'erreur
   	
    # Code de débogage pour trouver les lignes non numériques
	lignes_problematiques = df_ticker[~df_ticker['Close'].apply(lambda x: isinstance(x, (int, float)))]
	if not lignes_problematiques.empty:
		st.warning("Des données non numériques ont été trouvées dans la colonne 'Close' :")
		st.dataframe(lignes_problematiques)
	
	st.subheader(f"Cours — {ticker}")
	base_chart = plot_prices_with_actions(df_ticker)
	actions_series: pd.Series | None = None
	st.sidebar.markdown("---")
	show_forecast = st.sidebar.toggle("Afficher prédictions (futures)", value=True)
	auto_predict_years = st.sidebar.slider(
		"Durée de prédiction (années futures)", min_value=1, max_value=5, value=2, step=1
	)
	df_for_forecast = df_ticker
	if show_forecast:
		horizon_days = 252 * auto_predict_years
		fut = forecast_future_with_ets_seasonal(
			df_for_forecast,
			horizon_days=horizon_days
		)
		if fut is not None and not fut.empty:
			pred_chart = alt.Chart(fut.reset_index()).mark_line(color="#ff7f0e").encode(
				x=alt.X("index:T", title="Date"), y=alt.Y("PredictedClose:Q", title="Close"), tooltip=["index:T", "PredictedClose:Q"]
			).properties(height=500)
			base_chart = alt.layer(base_chart, pred_chart)
	st.altair_chart(base_chart, use_container_width=True)
	if run_button:
		# Le code ici ne s'exécutera que si df_ticker n'est pas vide
		with st.spinner("Simulation en cours..."):
			actions_series = sma_crossover_actions(df_ticker)
		result = simulate_portfolio(df_ticker, actions_series)
  
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
		# Courbe des actions et NetWorth
		st.altair_chart(plot_prices_with_actions(df_ticker, actions_series), use_container_width=True)
		nw_chart = alt.Chart(result.reset_index()).mark_line(color="#2ca02c").encode(
			x=alt.X("index:T", title="Date"), y=alt.Y("NetWorth:Q", title="Net Worth"), tooltip=["index:T", "NetWorth:Q"]
		).properties(height=400)
		st.altair_chart(nw_chart, use_container_width=True)


if __name__ == "__main__":
	main()
