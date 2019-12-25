import sys
from data import ImdbData
from experiment import Experiment

imdb_experiment = Experiment(ImdbData)
imdb_experiment.run()

if len(sys.argv) > 1:
    review_text = sys.argv[1]

    print('the sentiment of the review is: ', imdb_experiment.predict(review_text))
