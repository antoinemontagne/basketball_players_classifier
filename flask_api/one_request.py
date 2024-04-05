import requests

url_post_predict = "http://localhost:5000/predict"
one_player_data = {
  "GP": 47.0,
  "MIN": 11.7,
  "PTS": 4.4,
  "FGM": 1.6,
  "FGA": 4.4,
  "FG%": 36.9,
  "3P Made": 0.4,
  "3PA": 1.3,
  "3P%": 33.3,
  "FTM": 0.7,
  "FTA": 1.0,
  "FT%": 67.3,
  "OREB": 0.2,
  "DREB": 0.7,
  "REB": 0.9,
  "AST": 1.4,
  "STL": 0.7,
  "BLK": 0.1,
  "TOV": 0.9
}

response = requests.post(url_post_predict, json=one_player_data)

print(response.text)
