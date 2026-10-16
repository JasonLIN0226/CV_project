"""Software-only coupled continuation for the already proposed v5 deletion design.

No target, feature, label, reader, archive or oracle interface. This module is not
in any active/pending source lock and authorizes no scientific diagnostic.
"""
from v5_retention import deletion_forks, insert


def common_reservoir_continuations(state, incoming, draws, current_step,
                                   episode_length=256, recent=8, steps=32):
    """All 17 legal forks followed by exactly the same 32 Reservoir draws.

    State zero is the B+1-to-B deletion at current_step. Each subsequent state
    admits only the next chronological identity exiting the recent window.
    Deleted observations cannot be read from an archive and inserted again.
    Every continuation step must retain all four registered horizons; even the
    last step's h64 target must be inside the same episode. Never truncate the
    common cost or substitute a shorter horizon near an episode boundary.
    """
    draws = tuple(draws)
    if (state.capacity != 16 or episode_length != 256 or recent != 8
            or steps != 32 or len(draws) != steps
            or current_step - recent != incoming
            or current_step < 0 or current_step + steps + 64 >= episode_length):
        raise ValueError('registered continuation/time boundary changed')
    trajectories = []
    for deleted, fork in deletion_forks(state, incoming):
        trajectory = [fork]
        discarded = {deleted}
        for offset, draw in enumerate(draws, 1):
            fork, dropped = insert(fork, incoming + offset, 'Reservoir', draw)
            if dropped is not None:
                discarded.add(dropped)
            if discarded.intersection(fork.retained):
                raise ValueError('deleted observation re-entered memory')
            if len(fork.retained) != 16:
                raise ValueError('unequal capacity during continuation')
            trajectory.append(fork)
        trajectories.append((deleted, tuple(trajectory)))
    return tuple(trajectories)
