from collections import defaultdict

class POMDP:

    def __init__(self, state_weights_file, state_transition_file, observation_file):

        self.states = []
        self.actions_boolean = self.actions_detected(state_transition_file)
        self.initial_probabilities = self.read_state_probabilities(state_weights_file)
        self.transition_probabilities = self.read_transition_probabilities(state_transition_file)
        self.observation_probabilities = self.read_observation_probabilities(observation_file)

    def actions_detected(self, filename):
        with open(filename, 'r') as f:
            f.readline()
            f.readline()
            the_line = f.readline().strip()
            return len(the_line.split()) == 4

    def read_state_probabilities(self, filename):

        initial_probabilities = {}
        with open(filename, 'r') as f:
            f.readline()
            the_parts = f.readline().split()
            print("PARTS", the_parts)
            if len(the_parts) == 2:
                _, default_weight = map(int, the_parts)
            elif len(the_parts) == 1:
                default_weight = map(int, the_parts)

            total_weight = 0
            for line in f:
                state, weight = line.split()
                weight = int(weight)
                initial_probabilities[state.strip('"')] = weight
                total_weight += weight

            for state in initial_probabilities:
                initial_probabilities[state] /= total_weight
        self.states = list(initial_probabilities.keys())
        return initial_probabilities

    def read_transition_probabilities(self, filename):
        if self.actions_boolean:
            # Little Prince scenario
            transitions = defaultdict(lambda: defaultdict(dict))
            with open(filename, 'r') as f:
                f.readline()
                _, num_states, num_actions, default_weight = map(int, f.readline().split())
                total_weights = defaultdict(lambda: defaultdict(int))

                # Read specified transitions
                for line in f:
                    s1, action, s2, weight = line.split()
                    s1, action, s2 = s1.strip('"'), action.strip('"'), s2.strip('"')
                    weight = int(weight)
                    transitions[s1][action][s2] = weight
                    total_weights[s1][action] += weight

                # Add default weights for unspecified transitions
                for s1 in self.states:
                    for action in ["Forward", "Backward", "Turnaround"]:
                        for s2 in self.states:
                            if s2 not in transitions[s1][action]:
                                transitions[s1][action][s2] = default_weight
                                total_weights[s1][action] += default_weight

                # Create probabilites for each transition
                for s1 in transitions:
                    for action in transitions[s1]:
                        for s2 in transitions[s1][action]:
                            if not total_weights[s1][action] == 0:
                                transitions[s1][action][s2] /= total_weights[s1][action]
                                print("Transition Table:", s1, action, s2, transitions[s1][action][s2])

            return transitions
        else:
            # Speech Recognition scenario
            transitions = defaultdict(dict)
            with open(filename, 'r') as f:
                f.readline()
                _, num_states, num_actions, default_weight = map(int, f.readline().split())
                total_weights = defaultdict(int)

                # Transitions
                for line in f:
                    s1, s2, weight = line.split()
                    s1, s2 = s1.strip('"'), s2.strip('"')
                    weight = int(weight)
                    transitions[s1][s2] = weight
                    total_weights[s1] += weight

                # If transition not specified, give default weight to transition
                for s1 in self.states:
                    for s2 in self.states:
                        if s2 not in transitions[s1]:
                            transitions[s1][s2] = default_weight
                            total_weights[s1] += default_weight

                # Create probabilities for each transition
                for s1 in transitions:
                    for s2 in transitions[s1]:
                        if not total_weights[s1] == 0:
                            transitions[s1][s2] /= total_weights[s1]

            return transitions

    def read_observation_probabilities(self, filename):
        observations = defaultdict(lambda: defaultdict(float))
        with open(filename, 'r') as f:
            f.readline()
            _, num_states, num_observations, default_weight = map(int, f.readline().split())
            total_weights = defaultdict(int)

            for line in f:
                state, obs, weight = line.split()
                state, obs = state.strip('"'), obs.strip('"')
                weight = int(weight)
                observations[state][obs] = weight
                total_weights[state] += weight

            # If transition not specified, give default weight to transition
            for state in self.states:
                for obs in ["S", "Z", "EH0", "AH0"]:
                    if obs not in observations[state]:
                        observations[state][obs] = default_weight
                        total_weights[state] += default_weight

            # Create probabilities for each state observation
            for state in observations:
                for obs in observations[state]:
                    if not total_weights[state] == 0:
                        observations[state][obs] /= total_weights[state]

        return observations

    def viterbi(self, observation_sequence, action_sequence=None):
        obs_length = len(observation_sequence)
        Viterbi = [{}]
        path = {}
        last_path = None

        for state in self.states:
            initial_probability = self.initial_probabilities.get(state, 0)
            observation_probability = self.observation_probabilities[state].get(observation_sequence[0], 0)
            Viterbi[0][state] = initial_probability * observation_probability
            path[state] = [state]

        for time in range(1, obs_length):
            Viterbi.append({})
            new_path = {}
            print("VITERBI", Viterbi)

            for state2 in self.states:
                max_probability = 0
                best_state = None

                for state1 in self.states:
                    prev_prob = Viterbi[time - 1].get(state1, 0)
                    if self.actions_boolean:
                        # Little Prince
                        action_key = action_sequence[time - 1].strip()

                        trans_prob = self.transition_probabilities[state1].get(action_key, {}).get(state2, 0)
                    else:
                        # Speech Recognition
                        trans_prob = self.transition_probabilities[state1].get(state2, 0)
                    obs_prob = self.observation_probabilities[state2].get(observation_sequence[time], 0)
                    prob = prev_prob * trans_prob * obs_prob

                    print(
                        f"Time {time}, From {state1} to {state2}: Prev Prob={prev_prob}, Trans Prob={trans_prob}, "
                        f"Obs Prob={obs_prob}, Combined Prob={prob}")

                    if prob > max_probability:
                        max_probability = prob
                        best_state = state1

                Viterbi[time][state2] = max_probability
                if best_state is not None:
                    new_path[state2] = path[best_state] + [state2]

            if new_path:
                last_path = new_path
                print("LAST VALID PATH: ", last_path)

            path = new_path
            print("PATH:", path)

        max_probability = float('-inf')
        final_state = None

        for s in self.states:
            prob = Viterbi[obs_length - 1][s]
            if prob > max_probability:
                max_probability = prob
                final_state = s

        if final_state in path:
            return path[final_state]
        else:
            print(f"Error: Final state '{final_state}' not found in path.")
            return list(last_path.values())[0] if last_path else []


if __name__ == "__main__":

    state_weights_file = 'state_weights.txt'
    state_transition_file = 'state_action_state_weights.txt'
    observation_file = 'state_observation_weights.txt'
    observation_action_file = 'observation_actions.txt'

    solver = POMDP(state_weights_file, state_transition_file, observation_file)

    observations = []
    actions = []

    with open(observation_action_file, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.split()
            if len(parts) == 2:

                obs, act = parts
                observations.append(obs.strip('"'))
                actions.append(act.strip('"'))
            elif len(parts) == 1:

                obs = parts[0]
                observations.append(obs.strip('"'))
                actions.append("N")
    print("THE OBSERVATIONS:", observations)

    state_sequence = solver.viterbi(observations, actions)

    with open('states.txt', 'w') as f:
        f.write(f"states\n{len(state_sequence)}\n")
        for state in state_sequence:
            f.write(f'"{state}"\n')

