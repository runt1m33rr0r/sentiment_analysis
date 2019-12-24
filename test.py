from utils import ImdbData
from time import sleep

imdb_data = ImdbData(10_000)
print(len(imdb_data.texts), len(imdb_data.labels))

sleep(20)
