import time
import csv
import sys

from greedy_spanner import greedy_spanner
from yao import yao_graph
from theta import theta_graph
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
GRAPHS["Train stations LUX"] = Graph.datapoints_graph([v for k, v in train_stations_by_label_luxembourg.iteritems() if v is not None])
#Our own data challenge
GRAPHS["Our data challenge"] = Graph.from_data_challenge("data/data challenge.txt")
for n in range(1,7): #(1..6)
    GRAPHS["Data challenge %d" % n] = Graph.from_data_challenge("data/B%d.txt" % n)[0]

# Algorithms
ALGOS = {"Greedy": greedy_spanner, "Yao": yao_graph, "Theta": theta_graph}
ALGO_MAX_SIZE = {"Greedy": 350, "Yao": 3000, "Theta": 3000}

# Dilation ratios
RATIOS = (1.3, 1.5, 1.75, 2, 2.5, 3, 4, 5)

sys.stderr = open("stderr.txt", "a")
LOG_FILE = "experiment_log.csv"

with open(LOG_FILE, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("Graph", "#Vertices", "Algorithm", "Required dilation ratio", "Actual dilation ratio", "#Edges", "Total edge weight", "Max edge degree", "Diameter", "Running time"))

for g_name, g in GRAPHS.iteritems():
    for algo_name, algo in ALGOS.iteritems():
        if g.n_vertices() > ALGO_MAX_SIZE[algo_name]:
            print "Skipping algorithm %s for graph %s" % (algo_name, g_name)
            continue
            
        for ratio in RATIOS:
            t_start = time.clock()
            algo(g, ratio)
            t_elapsed = time.clock() - t_start
            
            with open(LOG_FILE, 'a') as f:
                writer = csv.writer(f)
                writer.writerow((g_name, g.n_vertices(), algo_name, ratio, g.dilation_ratio(), g.n_edges(), g.weight(), g.max_edge_degree(), g.diameter(), t_elapsed))
            
            g.clear_edges()
            print "Completed", g_name, algo_name, ratio