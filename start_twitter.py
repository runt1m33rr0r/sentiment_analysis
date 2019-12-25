import sys
from data import TwitterData
from experiment import Experiment

twitter_experiment = Experiment(TwitterData, 'twitter_nn', should_load_model=True)
twitter_experiment.run()

if len(sys.argv) > 1:
    tweet = sys.argv[1]
    print('the sentiment of the tweet is: ', twitter_experiment.predict(tweet))
