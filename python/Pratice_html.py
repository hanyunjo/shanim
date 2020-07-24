from bs4 import BeautifulSoup as bs
import requests
from Games import Game

url = "https://play.google.com/store/apps/collection/cluster?clp=0g4cChoKFHRvcHNlbGxpbmdfcGFpZF9HQU1FEAcYAw%3D%3D:S:ANO1ljLtt38&gsr=Ch_SDhwKGgoUdG9wc2VsbGluZ19wYWlkX0dBTUUQBxgD:S:ANO1ljJCqyI"
res = requests.get(url)
soup = bs(res.text, 'html.parser')

games = []
card_list = soup.select('div.Ktdaqe') # Ktdaqe = card-list

for i in card_list:
    cards = i.select('.ImZGtf') # ImZGtf = card
    for c in cards:
        games.append(Game(c))

for i in games:
    print(i)

with open("games_rank.csv", "w", encoding='utf-8') as file:
    for i in games:
        file.write(i.to_str() + "\n")