# A* Graph Search (g-value tracking) — README

What this script does
- Implements A* Graph Search with g-value tracking to avoid expanding paths that are worse than already-known best costs to the same node.
- Uses a priority queue of tuples (f, g, path) and a dictionary `g_valueTracker` that stores the best g-value seen for each node.
- Prints the found path and its cost when the goal node is popped from the heap with an optimal (or at least non-worse) g-value.

Dependencies
- Python 3.7+
- numpy
- heapq (standard library)

How the code is organized
- V: list of tuples (node_name, heuristic_value)
- E: list of weighted directed edges as (from_node_name, to_node_name, cost)
- Builds adjacency matrix `adj` of costs.
- Maintains `g_valueTracker` dictionary mapping node_index -> best_g_value so far.
- When a popped path has gVal > g_valueTracker[currentNode], it is skipped (pruned).

How to run
- Save the script as `astar_graph.py` and run:
  ```bash
  python astar_graph.py
  ```
- The graph is hard-coded; no interactive input needed.
- Example output:
  - "Found the goal node."
  - A printed path (sequence of node names).
  - "Cost obtained finally is this = X"

