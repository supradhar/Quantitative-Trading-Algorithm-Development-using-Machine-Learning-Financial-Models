import yfinance as yf
data = yf.download("MSFT", period="1y", interval="1d")
print(data)