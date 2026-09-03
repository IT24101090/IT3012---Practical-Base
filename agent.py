from collections import deque
import heapq
import math


class SearchAgent:

  def __init__(self, active_algo="AStar"):
    self.plan = []
    self.active_algo = active_algo  # 'BFS', 'DFS', 'UCS', or 'AStar'
    self.actions = [
        ("Up", (0, 1)),
        ("Down", (0, -1)),
        ("Left", (-1, 0)),
        ("Right", (1, 0)),
    ]

  # --- Step 1.1: Heuristic Functions ---
  def manhattan_distance(self, pos, goal):
    """Calculates h(n) = |x1 - x2| + |y1 - y2|"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

  def euclidean_distance(self, pos, goal):
    """Calculates h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)"""
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

  def get_neighbors(self, state, walls, grid_size):
    x, y = state
    width, height = grid_size
    neighbors = []
    for action, (dx, dy) in self.actions:
      nx, ny = x + dx, y + dy
      if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
        neighbors.append((action, (nx, ny)))
    return neighbors

  def reconstruct_path(self, parent_map, goal):
    path = []
    curr = goal
    while curr in parent_map and parent_map[curr] is not None:
      prev_state, action = parent_map[curr]
      path.append(action)
      curr = prev_state
    path.reverse()
    return path

  # --- Existing Search Algorithms ---
  def bfs_search(self, start, goal, walls, grid_size):
    frontier = deque([start])
    reached = {start}
    parent_map = {start: None}

    while frontier:
      curr = frontier.popleft()
      if curr == goal:
        return self.reconstruct_path(parent_map, goal)

      for action, neighbor in self.get_neighbors(curr, walls, grid_size):
        if neighbor not in reached:
          reached.add(neighbor)
          parent_map[neighbor] = (curr, action)
          frontier.append(neighbor)
    return []

  def dfs_search(self, start, goal, walls, grid_size):
    frontier = [start]
    reached = {start}
    parent_map = {start: None}

    while frontier:
      curr = frontier.pop()
      if curr == goal:
        return self.reconstruct_path(parent_map, goal)

      for action, neighbor in self.get_neighbors(curr, walls, grid_size):
        if neighbor not in reached:
          reached.add(neighbor)
          parent_map[neighbor] = (curr, action)
          frontier.append(neighbor)
    return []

  def ucs_search(self, start, goal, walls, grid_size):
    counter = 0
    frontier = [(0, counter, start)]
    reached = {start: 0}
    parent_map = {start: None}

    while frontier:
      cost, _, curr = heapq.heappop(frontier)

      if curr == goal:
        return self.reconstruct_path(parent_map, goal)

      for action, neighbor in self.get_neighbors(curr, walls, grid_size):
        new_cost = cost + 1
        if neighbor not in reached or new_cost < reached[neighbor]:
          reached[neighbor] = new_cost
          parent_map[neighbor] = (curr, action)
          counter += 1
          heapq.heappush(frontier, (new_cost, counter, neighbor))
    return []

  # --- Step 1.2: A* Search Algorithm ---
  def astar_search(
      self, start_pos, goal_pos, walls, grid_size, heuristic_type="manhattan"
  ):
    # Select heuristic function based on input parameter
    if heuristic_type == "euclidean":
      h_func = self.euclidean_distance
    else:
      h_func = self.manhattan_distance

    # Priority queue tuple structure: (f_cost, g_cost, counter, current_pos, path_taken)
    counter = 0
    g_start = 0
    h_start = h_func(start_pos, goal_pos)
    f_start = g_start + h_start

    frontier = [(f_start, g_start, counter, start_pos, [])]
    reached_states = set()

    while frontier:
      f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)

      if current_pos == goal_pos:
        return path_taken

      if current_pos in reached_states:
        continue

      reached_states.add(current_pos)

      for action, neighbor in self.get_neighbors(current_pos, walls, grid_size):
        if neighbor not in reached_states:
          g_new = g_cost + 1
          h_new = h_func(neighbor, goal_pos)
          f_new = g_new + h_new
          counter += 1
          heapq.heappush(
              frontier, (f_new, g_new, counter, neighbor, path_taken + [action])
          )

    return []

  # --- Step 1.3: Agent Perception & Decision Loop ---
  def sense_and_act(self, percept: dict) -> str:
    if not self.plan:
      start = tuple(percept["agent_pos"])
      all_food = percept.get(
          "all_food", percept.get("food_positions", percept.get("remaining_food", []))
      )

      if not all_food:
        return "Up"  # Fallback if no food remains

      # Select target food based on closest Manhattan distance
      goal = min(
          all_food, key=lambda f: abs(start[0] - f[0]) + abs(start[1] - f[1])
      )
      walls = set(tuple(w) for w in percept["walls"])
      grid_size = percept["grid_size"]

      if self.active_algo == "BFS":
        self.plan = self.bfs_search(start, goal, walls, grid_size)
      elif self.active_algo == "DFS":
        self.plan = self.dfs_search(start, goal, walls, grid_size)
      elif self.active_algo == "UCS":
        self.plan = self.ucs_search(start, goal, walls, grid_size)
      elif self.active_algo == "AStar":
        self.plan = self.astar_search(
            start, goal, walls, grid_size, heuristic_type="manhattan"
        )

    if self.plan:
      return self.plan.pop(0)
    return "Up"


# Testing Checkpoint for Step 1.1
if __name__ == "__main__":
  agent = SearchAgent()
  mock_start = (0, 0)
  mock_goal = (3, 4)

  print("Step 1.1 Testing Checkpoint:")
  print(f"Manhattan Distance: {agent.manhattan_distance(mock_start, mock_goal)}")
  print(f"Euclidean Distance: {agent.euclidean_distance(mock_start, mock_goal)}")