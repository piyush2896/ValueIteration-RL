from collections import defaultdict

def get_max_v_a_pair(world,
                     v_star,
                     state,
                     actions,
                     noise,
                     gamma):
    v_a_pairs = []
    for action in actions:
        is_end, n_state = world.move_given_action(state, action)
        if is_end:
            v = n_state
        else:
            lr_states = world.move_lr_given_action(state, action)
            v_n_state = (1. - noise) * (world.get_reward(state, action, n_state) +
                v_star[n_state] * gamma)
            v_l_state = (noise / 2) * (world.get_reward(state, action, lr_states[0]) +
                v_star[lr_states[0]] * gamma)
            v_r_state = (noise / 2) * (world.get_reward(state, action, lr_states[1]) +
                v_star[lr_states[1]] * gamma)
            v = v_n_state + v_l_state + v_r_state

        v_a_pairs.append((v, action))
    return max(v_a_pairs)

def value_iter(world,
               noise=0.2,
               gamma=0.99,
               h=100,
               v_star_init=None,
               verbose=5):
    if v_star_init is not None:
        v_star = v_star_init
    else:
        v_star = defaultdict(lambda: 0.)
    pi_star = {}
    for state in world.pos_reward_states:
        pi_star[state] = world.EXIT
    for state in world.neg_reward_states:
        pi_star[state] = world.EXIT

    for k in range(1, h+1):
        for state in world.states:
            possible_actions = world.actions_available(state)
            v_star[state], pi_star[state] = get_max_v_a_pair(world,
                v_star, state, possible_actions, noise, gamma)
        if k % verbose == 0:
            print('Horizon {}/{}'.format(k, h), '-'*30)
            world.display_world_v_vals(v_star)
            print('\n')
    return v_star, pi_star
