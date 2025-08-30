def solve_electric_car(distance, initial_battery):
    from collections import deque
    
    queue = deque([(0, 0, initial_battery, 0)])
    visited = {}
    min_time = float('inf')
    max_battery_at_min_time = 0
    
    while queue:
        time, dist_covered, battery, consecutive_moves = queue.popleft()
        
        if dist_covered == distance:
            if time < min_time:
                min_time = time
                max_battery_at_min_time = battery
            elif time == min_time:
                max_battery_at_min_time = max(max_battery_at_min_time, battery)
            continue
        
        state = (dist_covered, battery, consecutive_moves)
        if state in visited and visited[state] <= time:
            continue
        visited[state] = time
        
        if time >= min_time:
            continue
        
        if time + 1 < min_time:
            queue.append((time + 1, dist_covered, battery + 1, 0))
        
        next_move_cost = consecutive_moves + 1
        if battery >= next_move_cost and dist_covered < distance and time + 1 < min_time:
            queue.append((time + 1, dist_covered + 1, battery - next_move_cost, consecutive_moves + 1))
    
    return min_time, max_battery_at_min_time

def main():
    n = int(input())
    
    for _ in range(n):
        distance, initial_battery = map(int, input().split())
        time, battery = solve_electric_car(distance, initial_battery)
        print(time, battery)

if __name__ == "__main__":
    main()