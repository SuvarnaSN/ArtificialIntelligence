# A* Tree Search — README

What this script does
- Implements A* Tree Search on a directed graph with non-negative edge costs and node heuristics.
- The implementation treats paths in the heap independently (no closed-set or g-value pruning), so multiple paths to the same node can be expanded.
- Prints the first path that reaches the goal and its cost.

Dependencies
- Python 3.7+
- numpy
- heapq (standard library)

How the code is organized
- V: list of tuples (node_name, heuristic_value)
- E: list of weighted directed edges as (from_node_name, to_node_name, cost)
- Builds an adjacency matrix of edge costs.
- Uses a priority queue of tuples (f, g, path) where f = g + h(next_node).
- Pops the best f each iteration and expands children, pushing new (f, g, path) tuples.

How to run
- Save the script to a file and run:
  ```bash
  python astar_tree.py
  ```
- The graph is hard-coded. No interactive input required.
- Example printed output:
  - "Found the goal node."
  - Path printed like: `S -> D -> B -> E -> G`
  - "Cost obtained finally is this = 10.0"

Important behavior / limitations
- This is the A* Tree Search variant: it does not prevent revisiting nodes via different paths. That may cause large memory/expansion for graphs with cycles or many alternative routes.
- The algorithm returns the first time it pops a path whose current node is the goal — this is correct for tree search but may be inefficient for graphs with repeated states.

Known issues & suggestions
- If you want optimality on graphs with repeated states and admissible heuristics, use the A* Graph Search variant (with a closed set or g-value tracking).
- For large graphs, replace adjacency matrix with adjacency lists to improve performance and reduce memory.
- Consider storing parent pointers and reconstructing the path rather than storing full path lists in the heap (less memory pressure).
- Add checks for heuristic admissibility if optimal path cost is required.

