import random

from models.tick_markov import TickMarkovModel


def test_abstains_before_min_context_count():
    model = TickMarkovModel(max_order=2)
    for _ in range(5):
        model.push(0.001)
    assert model.vote() == 0.0


def test_learns_a_biased_up_sequence():
    random.seed(0)
    model = TickMarkovModel(max_order=1)
    # 80% up-ticks, unconditional -- order-1 context should converge near +0.6
    for _ in range(500):
        lr = 0.001 if random.random() < 0.8 else -0.001
        model.push(lr)
    vote = model.vote()
    assert vote > 0.4, f"expected a strong up-lean vote, got {vote}"


def test_symmetric_random_walk_votes_near_zero():
    random.seed(1)
    model = TickMarkovModel(max_order=1)
    for _ in range(1000):
        lr = 0.001 if random.random() < 0.5 else -0.001
        model.push(lr)
    assert abs(model.vote()) < 0.15


def test_backs_off_to_lower_order_when_high_order_context_is_rare():
    model = TickMarkovModel(max_order=2)
    # order-1 gets plenty of support; a specific order-2 context never
    # recurs often enough to clear MIN_CONTEXT_COUNT on its own.
    random.seed(2)
    for _ in range(200):
        lr = 0.001 if random.random() < 0.7 else -0.001
        model.push(lr)
    # should not abstain -- order-1 has enough support even if some
    # specific order-2 context doesn't
    assert model.vote() != 0.0 or model.sample_size < TickMarkovModel.MIN_CONTEXT_COUNT
