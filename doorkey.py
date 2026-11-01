from utils import *
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import argparse
import os

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

# direction vectors: 0=right, 1=down, 2=left, 3=up
DIR_TO_VEC = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}


def get_next_state(state, action, info):
    """
    state = (col, row, direction, has_key, door_open)
    Returns (next_state, cost)
    """
    col, row, direction, has_key, door_open = state
    cost = step_cost(action)

    if action == TL:
        direction = (direction - 1) % 4

    elif action == TR:
        direction = (direction + 1) % 4

    elif action == MF:
        front = np.array([col, row]) + DIR_TO_VEC[direction]

        if not (0 <= front[0] < info["width"] and 0 <= front[1] < info["height"]):
            return (col, row, direction, has_key, door_open), cost

        if (front[0], front[1]) in info["walls"]:
            return (col, row, direction, has_key, door_open), cost

        if np.array_equal(front, info["door_pos"]) and not door_open:
            return (col, row, direction, has_key, door_open), cost

        col, row = int(front[0]), int(front[1])

    elif action == PK:
        front = np.array([col, row]) + DIR_TO_VEC[direction]
        if np.array_equal(front, info["key_pos"]) and not has_key:
            has_key = 1

    elif action == UD:
        front = np.array([col, row]) + DIR_TO_VEC[direction]
        if np.array_equal(front, info["door_pos"]) and has_key:
            door_open = 1

    return (col, row, direction, has_key, door_open), cost

def build_walls(env, info):
    walls = set()
    for i in range(info["height"]):
        for j in range(info["width"]):
            cell = env.unwrapped.grid.get(j, i)
            if cell is not None and cell.type == "wall":
                walls.add((j, i))
    info["walls"] = walls


def dp_policy(env, info):
    """
    Returns optimal action sequence from initial state to goal.
    """
    build_walls(env, info)

    all_states = []
    for col in range(info["width"]):
        for row in range(info["height"]):
            if (col, row) in info["walls"]:
                continue
            for direction in range(4):
                for has_key in [0, 1]:
                    for door_open in [0, 1]:
                        all_states.append((col, row, direction, has_key, door_open))

    V = {}
    for state in all_states:
        col, row, direction, has_key, door_open = state
        if (col, row) == tuple(info["goal_pos"]):
            V[state] = 0
        else:
            V[state] = float("inf")

    # Value iteration
    actions = [MF, TL, TR, PK, UD]
    while True:
        delta = 0
        for state in all_states:
            col, row, direction, has_key, door_open = state
            if (col, row) == tuple(info["goal_pos"]):
                continue
            best = float("inf")
            for action in actions:
                next_state, cost = get_next_state(state, action, info)
                if V[next_state] != float("inf"):
                    best = min(best, cost + V[next_state])
            delta = max(delta, abs(best - V[state]))
            V[state] = best
        if delta < 1e-6:
            break

    #Extract policy
    policy = {}
    for state in all_states:
        col, row, direction, has_key, door_open = state
        if (col, row) == tuple(info["goal_pos"]):
            continue
        best_action = None
        best_val = float("inf")
        for action in actions:
            next_state, cost = get_next_state(state, action, info)
            val = cost + V[next_state]
            if val < best_val:
                best_val = val
                best_action = action
        policy[state] = best_action

    # Simulate from initial state
    dir_vec_to_idx = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    init_dir = dir_vec_to_idx[tuple(info["init_agent_dir"])]
    init_col, init_row = int(info["init_agent_pos"][0]), int(info["init_agent_pos"][1])

    state = (init_col, init_row, init_dir, 0, 0)
    seq = []
    while (state[0], state[1]) != tuple(info["goal_pos"]):
        action = policy[state]
        seq.append(action)
        state, _ = get_next_state(state, action, info)

    return seq

def doorkey_problem(env, info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env
    """
    return dp_policy(env, info)


def partA():
    os.makedirs("gif", exist_ok=True)
    env_paths = [
        "./envs/known_envs/doorkey-5x5-normal.env",
        "./envs/known_envs/doorkey-6x6-normal.env",
        "./envs/known_envs/doorkey-8x8-normal.env",
        "./envs/known_envs/doorkey-6x6-direct.env",
        "./envs/known_envs/doorkey-8x8-direct.env",
        "./envs/known_envs/doorkey-6x6-shortcut.env",
        "./envs/known_envs/doorkey-8x8-shortcut.env",
    ]
    for env_path in env_paths:
        env, info = load_env(env_path)
        seq = doorkey_problem(env, info)
        total_cost = sum(step_cost(a) for a in seq)
        env_name = os.path.splitext(os.path.basename(env_path))[0]
        print(f"{env_name}: {len(seq)} steps, total cost = {total_cost}")
        draw_gif_from_seq(seq, load_env(env_path)[0], path=f"./gif/{env_name}.gif")


KEY_POSITIONS  = [(2, 2), (2, 3), (1, 6)]
GOAL_POSITIONS = [(6, 1), (7, 3), (6, 6)]
DOOR_POSITIONS = [(5, 3), (5, 7)]


def get_next_state_partB(state, action, walls):
    """
    state = (col, row, direction, has_key, door1_open, door2_open, key_idx, goal_idx)
    Returns (next_state, cost)
    """
    col, row, direction, has_key, door1_open, door2_open, key_idx, goal_idx = state
    cost = step_cost(action)
    door_open = [door1_open, door2_open]

    if action == TL:
        direction = (direction - 1) % 4

    elif action == TR:
        direction = (direction + 1) % 4

    elif action == MF:
        front = np.array([col, row]) + DIR_TO_VEC[direction]
        fx, fy = int(front[0]), int(front[1])

        if not (0 <= fx < 10 and 0 <= fy < 10):
            return state, cost
        if (fx, fy) in walls:
            return state, cost
        # check each door
        blocked = False
        for i, dpos in enumerate(DOOR_POSITIONS):
            if (fx, fy) == dpos and not door_open[i]:
                blocked = True
                break
        if blocked:
            return state, cost
        col, row = fx, fy

    elif action == PK:
        front = np.array([col, row]) + DIR_TO_VEC[direction]
        fx, fy = int(front[0]), int(front[1])
        if not has_key and key_idx < len(KEY_POSITIONS) and (fx, fy) == KEY_POSITIONS[key_idx]:
            has_key = 1

    elif action == UD:
        front = np.array([col, row]) + DIR_TO_VEC[direction]
        fx, fy = int(front[0]), int(front[1])
        if has_key:
            for i, dpos in enumerate(DOOR_POSITIONS):
                if (fx, fy) == dpos:
                    door_open[i] = 1
                    break

    return (col, row, direction, has_key, door_open[0], door_open[1], key_idx, goal_idx), cost


def dp_policy_partB(walls):
    """
    Returns a policy dict mapping state -> action for all Part B configurations.
    state = (col, row, direction, has_key, door1_open, door2_open, key_idx, goal_idx)
    """
    # Enumerate all states
    all_states = []
    for col in range(10):
        for row in range(10):
            if (col, row) in walls:
                continue
            for direction in range(4):
                for has_key in [0, 1]:
                    for door1_open in [0, 1]:
                        for door2_open in [0, 1]:
                            for key_idx in range(len(KEY_POSITIONS)):
                                for goal_idx in range(len(GOAL_POSITIONS)):
                                    all_states.append((col, row, direction, has_key,
                                                       door1_open, door2_open,
                                                       key_idx, goal_idx))

    # Initialize V
    V = {}
    for state in all_states:
        col, row = state[0], state[1]
        goal_idx = state[7]
        if (col, row) == GOAL_POSITIONS[goal_idx]:
            V[state] = 0
        else:
            V[state] = float("inf")

    # Value iteration
    actions = [MF, TL, TR, PK, UD]
    while True:
        delta = 0
        for state in all_states:
            col, row, _, _, _, _, _, goal_idx = state
            if (col, row) == GOAL_POSITIONS[goal_idx]:
                continue
            best = float("inf")
            for action in actions:
                next_state, cost = get_next_state_partB(state, action, walls)
                if V[next_state] != float("inf"):
                    best = min(best, cost + V[next_state])
            delta = max(delta, abs(best - V[state]))
            V[state] = best
        if delta < 1e-6:
            break

    # Extract policy
    policy = {}
    for state in all_states:
        col, row, _, _, _, _, _, goal_idx = state
        if (col, row) == GOAL_POSITIONS[goal_idx]:
            continue
        best_action, best_val = None, float("inf")
        for action in actions:
            next_state, cost = get_next_state_partB(state, action, walls)
            val = cost + V[next_state]
            if val < best_val:
                best_val, best_action = val, action
        policy[state] = best_action

    return policy


def load_env_partB(env_path):
    """Load a random-map env and return (env, key_idx, goal_idx, door1_open, door2_open, init_state)."""
    import pickle
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    from minigrid.core.world_object import Key, Door, Goal
    key_pos, goal_pos, door_info = None, None, {}
    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            cell = env.unwrapped.grid.get(j, i)
            if isinstance(cell, Key):    key_pos  = (j, i)
            elif isinstance(cell, Goal): goal_pos = (j, i)
            elif isinstance(cell, Door): door_info[(j, i)] = cell.is_open

    key_idx  = KEY_POSITIONS.index(key_pos)
    goal_idx = GOAL_POSITIONS.index(goal_pos)
    door1_open = 1 if door_info.get(DOOR_POSITIONS[0], False) else 0
    door2_open = 1 if door_info.get(DOOR_POSITIONS[1], False) else 0

    dir_vec_to_idx = {(1,0):0, (0,1):1, (-1,0):2, (0,-1):3}
    init_dir = dir_vec_to_idx[tuple(env.unwrapped.dir_vec)]
    init_col, init_row = int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])
    init_state = (init_col, init_row, init_dir, 0, door1_open, door2_open, key_idx, goal_idx)

    return env, init_state, goal_idx


def partB():
    os.makedirs("gif", exist_ok=True)
    env_folder = "./envs/random_envs"

    # build walls from any env (structure is fixed for all 10x10 maps)
    env, _, _ = load_random_env(env_folder)
    walls = set()
    for i in range(10):
        for j in range(10):
            cell = env.unwrapped.grid.get(j, i)
            if cell is not None and cell.type == "wall":
                walls.add((j, i))

    # compute single policy for all configurations of Part B
    policy = dp_policy_partB(walls)

    # run policy on all 36 envs
    env_list = sorted([f for f in os.listdir(env_folder) if f.endswith(".env")])
    for env_file in env_list:
        env_path = os.path.join(env_folder, env_file)
        env, state, goal_idx = load_env_partB(env_path)

        # simulate policy
        seq = []
        while (state[0], state[1]) != GOAL_POSITIONS[goal_idx]:
            action = policy[state]
            seq.append(action)
            state, _ = get_next_state_partB(state, action, walls)

        total_cost = sum(step_cost(a) for a in seq)
        env_name = os.path.splitext(env_file)[0]
        draw_gif_from_seq(seq, env, path=f"./gif/{env_name}.gif")
        print(f"{env_file}: {len(seq)} steps, total cost = {total_cost}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Door-Key grids with dynamic programming")
    parser.add_argument("--part", choices=["known", "random", "all"], default="known")
    args = parser.parse_args()
    if args.part in ("known", "all"):
        partA()
    if args.part in ("random", "all"):
        partB()
