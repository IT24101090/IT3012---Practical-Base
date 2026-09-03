from collections import deque
import heapq

class SearchAgent:
    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo  # 'BFS', 'DFS', or 'UCS'
        self.actions = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

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
        # Priority queue entries: (cost, tie_breaker_counter, state)
        counter = 0
        frontier = [(0, counter, start)]
        reached = {start: 0}
        parent_map = {start: None}

        while frontier:
            cost, _, curr = heapq.heappop(frontier)

            if curr == goal:
                return self.reconstruct_path(parent_map, goal)

            for action, neighbor in self.get_neighbors(curr, walls, grid_size):
                new_cost = cost + 1  # Standard step cost = 1
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    parent_map[neighbor] = (curr, action)
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor))
        return []

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = tuple(percept['agent_pos'])
            all_food = percept.get('all_food', [])
            
            if not all_food:
                return 'Up'  # Fallback if no food remains

            # Select the target food (closest Manhattan distance)
            goal = min(all_food, key=lambda f: abs(start[0] - f[0]) + abs(start[1] - f[1]))
            walls = set(tuple(w) for w in percept['walls'])
            grid_size = percept['grid_size']

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, walls, grid_size)

        if self.plan:
            return self.plan.pop(0)
        return 'Up'