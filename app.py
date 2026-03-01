import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(
    page_title="Momentum Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# FUNZIONI LOGICHE (Business Logic)
# ==========================================

@st.cache_data(ttl=3600) # Cache i dati per 1 ora per velocizzare i reload
def load_data(tickers, start_date, end_date):
    """Scarica i dati storici usando yfinance con cache."""
    # Aggiungiamo QQQ se non presente
    if 'QQQ' not in tickers:
        tickers = tickers + ['QQQ']
    
    try:
        with st.spinner("Download dati in corso (potrebbero volerci alcuni secondi)..."):
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Errore durante il download dei dati: {e}")
        return None

def calculate_momentum_score(df):
    """Calcola il punteggio momentum composito (0-100)."""
    if len(df) < 130:
        return 0
    
    close = df['Close']
    volume = df['Volume']
    
    # 1. Price Momentum
    roc_20 = close.pct_change(20).iloc[-1]
    roc_60 = close.pct_change(60).iloc[-1]
    roc_126 = close.pct_change(126).iloc[-1]
    
    # 2. Volatility Adjusted
    returns = close.pct_change()
    vol = returns.rolling(20).std().iloc[-1]
    annualized_return = (1 + roc_20)**12 - 1
    annualized_vol = vol * np.sqrt(252)
    sharpe_proxy = (annualized_return / annualized_vol) if annualized_vol > 0 else 0
    
    # 3. Volume Confirmation
    vol_sma = volume.rolling(50).mean()
    vol_trend = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

    # 4. RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_score = rsi.iloc[-1]

    # Combinazione Ponderata
    raw_score = (
        (roc_126 * 100 * 0.5) + 
        (roc_60 * 100 * 0.3) + 
        (roc_20 * 100 * 0.1) + 
        (sharpe_proxy * 10 * 0.05) + 
        (np.clip(vol_trend, 0.5, 2.0) * 10 * 0.05)
    )
    
    normalized_score = 50 + (raw_score / 2) + (rsi_score / 10)
    return np.clip(normalized_score, 0, 100)

def get_qqq_trend(qqq_series, current_idx, ma_period=200):
    """Verifica se QQQ è sopra la media mobile all'indice specificato."""
    if current_idx < ma_period:
        return False
    current_price = qqq_series.iloc[current_idx]
    ma_200 = qqq_series.iloc[current_idx - ma_period : current_idx].mean()
    return current_price > ma_200

# ==========================================
# INTERFACCIA UTENTE
# ==========================================

st.title("📈 Momentum Strategy Backtester")
st.markdown("""
Simulazione di una strategia sistematica momentum basata su:
- **Ranking Giornaliero**: Punteggio multi-fattore.
- **Equal Weight**: Rebalancing giornaliero per mantenere pesi uguali.
- **Top N Selection**: Selezione dei migliori titoli.
- **QQQ Filter**: Filtro trend di mercato.
""")

# --- SIDEBAR CONFIGURAZIONE ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Input Ticker
    default_tickers = """AAPL
MSFT
GOOGL
AMZN
NVDA
META
TSLA
AMD
NFLX
INTC
ADBE
CRM
CSCO
ORCL
AVGO
QCOM
TXN
MRVL
MU
LRCX
SNAP
SHOP
SQ
COIN
RIVN
PLTR
"""
    tickers_input = st.text_area("Ticker Universe (uno per riga)", value=default_tickers, height=200)
    
    # Input Date
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data Inizio", datetime(2025, 1, 1))
    with col2:
        end_date = st.date_input("Data Fine", datetime.now())
        
    # Parametri Strategia
    st.subheader("Parametri Strategia")
    initial_capital = st.number_input("Capitale Iniziale ($)", value=1_000_000, step=100_000)
    top_n = st.slider("Top N Stocks (Posizioni)", min_value=5, max_value=50, value=10)
    min_score = st.slider("Soglia Minima Score", min_value=0, max_value=100, value=70)
    qqq_ma = st.number_input("QQQ MA Period", value=200)
    
    run_btn = st.button("🚀 Esegui Backtest", type="primary")

# ==========================================
# ESECUZIONE BACKTEST
# ==========================================

if run_btn:
    # 1. Preparazione Dati
    ticker_list = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
    
    if not ticker_list:
        st.warning("Inserisci almeno un ticker.")
    else:
        # Scarica dati
        raw_data = load_data(ticker_list, start_date, end_date)
        
        if raw_data is not None and not raw_data.empty:
            # Prepara i DataFrame
            price_data = raw_data['Adj Close']
            open_data = raw_data['Open']
            volume_data = raw_data['Volume']
            qqq_series = price_data['QQQ']
            
            if 'QQQ' in ticker_list:
                ticker_list.remove('QQQ')

            # 2. Loop di Backtesting
            st.info("Elaborazione backtest in corso...")
            
            # Variabili stato
            portfolio = {} 
            cash = initial_capital
            trades_log = []
            equity_history = []
            
            # Indici validi (dopo periodo MA)
            # Troviamo l'indice nel DataFrame che corrisponde alla data d'inizio + buffer
            # Per semplicità usiamo gli indici del DataFrame
            all_dates = price_data.index
            
            # Trova il primo indice dove abbiamo abbastanza dati per la MA200
            start_idx = qqq_ma + 130 # MA + dati per score
            
            if start_idx >= len(all_dates):
                st.error("Periodo selezionato troppo corto per calcolare MA200 e Score.")
            else:
                # Loop sugli indici
                for i in range(start_idx, len(all_dates)):
                    date = all_dates[i]
                    
                    # --- FASE 1: ANALISI (Giorno T) ---
                    is_uptrend = get_qqq_trend(qqq_series, i, qqq_ma)
                    
                    # Calcola Scores
                    current_scores = {}
                    for ticker in ticker_list:
                        # Controlla se il ticker esiste nei dati
                        if ticker not in price_data.columns: continue
                        
                        # Slice dati fino a oggi
                        hist_close = price_data[ticker].iloc[:i+1]
                        hist_vol = volume_data[ticker].iloc[:i+1]
                        
                        # Drop NaN
                        hist_close = hist_close.dropna()
                        hist_vol = hist_vol.dropna()
                        
                        if len(hist_close) < 130: continue
                        
                        df_t = pd.DataFrame({'Close': hist_close, 'Volume': hist_vol})
                        try:
                            score = calculate_momentum_score(df_t)
                            if score >= min_score:
                                current_scores[ticker] = score
                        except:
                            continue
                    
                    # Selezione Target
                    ranked = sorted(current_scores.items(), key=lambda item: item[1], reverse=True)
                    target_tickers = [t[0] for t in ranked[:top_n]]
                    
                    if not is_uptrend:
                        # In downtrend, teniamo solo quelli che abbiamo già e che sono ancora nel top
                        target_tickers = [t for t in target_tickers if t in portfolio.keys()]
                    
                    # --- FASE 2: ESECUZIONE (Simulata su T usando Open, o meglio T+1) ---
                    # Per semplicità nel loop singolo, usiamo Open del giorno T come prezzo di esecuzione
                    # In produzione reale bisognerebbe shiftare di 1.
                    
                    current_equity = cash
                    for t, shares in portfolio.items():
                        if t in price_data.columns:
                            price = price_data[t].iloc[i]
                            if pd.notna(price):
                                current_equity += shares * price
                    
                    equity_history.append({'Date': date, 'Equity': current_equity})
                    
                    # Calcolo target value
                    num_pos = len(target_tickers)
                    target_val = current_equity / num_pos if num_pos > 0 else 0
                    
                    # Esecuzione Trade
                    # 1. Exit
                    to_exit = set(portfolio.keys()) - set(target_tickers)
                    for ticker in to_exit:
                        shares = portfolio[ticker]
                        if ticker in open_data.columns:
                            price_exec = open_data[ticker].iloc[i]
                            if pd.notna(price_exec):
                                proceeds = shares * price_exec
                                cash += proceeds
                                trades_log.append({
                                    'symbol': ticker, 'Date': date, 'shares': shares,
                                    'price': round(price_exec, 2), 'value': round(proceeds, 2), 'type': 'Exit'
                                })
                                del portfolio[ticker]
                    
                    # 2. Rebalance / Entry
                    for ticker in target_tickers:
                        # Verifica dati
                        if ticker not in open_data.columns or ticker not in price_data.columns: continue
                        price_exec = open_data[ticker].iloc[i]
                        price_close = price_data[ticker].iloc[i]
                        if pd.isna(price_exec): continue

                        if ticker in portfolio:
                            # Rebalance
                            curr_shares = portfolio[ticker]
                            curr_val = curr_shares * price_close
                            diff = target_val - curr_val
                            
                            if abs(diff) > 100: # Soglia minima trade
                                shares_trade = int(diff / price_exec)
                                if shares_trade > 0:
                                    cost = shares_trade * price_exec
                                    if cash >= cost:
                                        cash -= cost
                                        portfolio[ticker] += shares_trade
                                        trades_log.append({'symbol': ticker, 'Date': date, 'shares': shares_trade, 'price': round(price_exec, 2), 'value': round(cost, 2), 'type': 'Increase'})
                                elif shares_trade < 0:
                                    cash += abs(shares_trade) * price_exec
                                    portfolio[ticker] += shares_trade
                                    trades_log.append({'symbol': ticker, 'Date': date, 'shares': abs(shares_trade), 'price': round(price_exec, 2), 'value': round(abs(shares_trade) * price_exec, 2), 'type': 'Decrease'})
                        else:
                            # Entry
                            shares_buy = int(target_val / price_exec)
                            if shares_buy > 0:
                                cost = shares_buy * price_exec
                                if cash >= cost:
                                    cash -= cost
                                    portfolio[ticker] = shares_buy
                                    trades_log.append({'symbol': ticker, 'Date': date, 'shares': shares_buy, 'price': round(price_exec, 2), 'value': round(cost, 2), 'type': 'Entry'})

                # ==========================================
                # VISUALIZZAZIONE RISULTATI
                # ==========================================
                
                # Conversione risultati
                df_equity = pd.DataFrame(equity_history)
                df_trades = pd.DataFrame(trades_log)
                
                if not df_equity.empty:
                    # Metriche
                    final_val = df_equity['Equity'].iloc[-1]
                    total_return = (final_val - initial_capital) / initial_capital * 100
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Valore Finale", f"${final_val:,.2f}")
                    col_m2.metric("Rendimento Totale", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
                    col_m3.metric("Numero Trades", len(df_trades))
                    col_m4.metric("Titoli Universo", len(ticker_list))
                    
                    # Grafico Equity
                    st.subheader("📊 Equity Curve")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_equity['Date'], y=df_equity['Equity'], mode='lines', name='Portfolio Value'))
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabella Trades
                    st.subheader("📝 Log Trades")
                    st.dataframe(df_trades, use_container_width=True)
                    
                    # Download CSV
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Scarica CSV Trades",
                        data=csv,
                        file_name='strategy_trades.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("Nessun dato sufficiente per generare il grafico.")