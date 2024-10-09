import os
import csv
import argparse
from pprint import pprint

ap = argparse.ArgumentParser()
ap.add_argument("model_dir", help="Directory of metrics.csv file")
args = ap.parse_args()

with open(os.path.join(args.model_dir, "metrics.csv"), 'r') as f:
    R = csv.reader(f)
    
    best = {}
    for line in R:
        if not line[1] in best:
            best[line[1]] = [line]
        else:
            best[line[1]].append(line)
    
    top_k = 3
    for key in best:
        final = []
        final.append([x for x in best[key] if x[3] == "1"])
        final.extend(sorted(best[key], key=lambda x: float(x[5]))[:top_k])
        best[key] = final

pprint (best)
