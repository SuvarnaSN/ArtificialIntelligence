# Greedy Best-First Search (GBFS) — README

What this script does
- Implements a Greedy Best-First Search (GBFS) over a small directed graph.
- Nodes are represented with heuristic values (h). The algorithm always expands the unexplored node with the smallest heuristic value.
- Produces a path (if any) from start `S` to goal `G` using parent pointers.

Dependencies
- Python 3.7+
- numpy

How the code is organized
- V: list of tuples (node_name, heuristic_value)
- E: list of directed edges as (from_node_name, to_node_name)
- Builds an adjacency matrix, runs GBFS using `min(nodesUnexplored, key=gettheHeuristicValue)` to pick next node.
- Records parent pointers and reconstructs path from `S` to `G` if reachable.

How to run
- Save the script to a .py file and run:
  ```bash
  python greedy_gbfs.py
  ```
- The script uses hard-coded graph (V and E). No interactive input is required.
- Prints edges, adjacency matrix, visited order (indices), parent list(s), and final path by node names.

Example output (conceptual)
- Visited node indices printed in exploration order
- "Goal node X found."
- Path printed as e.g. `['S', 'D', 'E', 'G']`
