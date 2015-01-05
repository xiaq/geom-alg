import time

from greedy_spanner import greedy_spanner
from datatypes import Graph

from data.train_stations import train_stations_by_label_luxembourg, train_stations_by_label_netherlands

# First generate the graphs
GRAPHS = {}
#Random graphs
for n in range(25, 351, 25):
    GRAPHS["Random %d" % n] = Graph.from_data_challenge("data/random %d.txt" % n)[0]
#Train stations NL
GRAPHS["Train stations NL"] = Graph.datapoints_graph([v for k, v in train_stations_by_label_netherlands.iteritems() if v is not None])
#Train stations Luxembourg
GRAPHS["Train stations LUX"] = Graph.datapoints_graph({k: v for k, v in train_stations_by_label_luxembourg.iteritems() if v is not None})
#Our own data challenge
GRAPHS["Data challenge"] = Graph.from_data_challenge("data/data challenge.txt")

# Algorithms
ALGOS = {"Greedy": greedy_spanner}

# Dilation ratios
RATIOS = (1.3, 1.5, 1.75, 2, 2.5, 3, 4, 5)

# TODO: Prepare log file
LOG_FILE = "experiment_log.txt"

for g_name, g in GRAPHS.iteritems():
    for algo_name, algo in ALGOS.iteritems():
        for ratio in RATIOS:
            t_start = time.clock()
            algo(g, ratio)
            t_elapsed = time.clock() - t_start
            
            # TODO Take statistics
            
            with open(LOG_FILE, 'a') as f:
                # TODO Write statistics to file
                pass
            
            g.clear()