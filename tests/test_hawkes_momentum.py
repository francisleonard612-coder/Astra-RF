import random

from models.hawkes_momentum import HawkesMomentumModel


def test_abstains_during_warmup():
    model = HawkesMomentumModel(warmup=30)
    for _ in range(20):
        model.push(0.0001)
    assert model.vote() == 0.0


def test_flags_upside_after_jump_cluster():
    random.seed(0)
    model = HawkesMomentumModel(decay=0.90, jump_multiple=2.0, warmup=30)
    # quiet baseline to establish a small volatility estimate
    for _ in range(40):
        model.push(random.gauss(0, 0.0001))
    # a cluster of large, same-direction jumps
    for _ in range(4):
        model.push(0.01)
    vote = model.vote()
    assert vote > 0.3, f"expected upside momentum vote, got {vote}"


def test_decays_back_toward_zero_after_jump_activity_stops():
    random.seed(1)
    model = HawkesMomentumModel(decay=0.80, jump_multiple=2.0, warmup=30)
    for _ in range(40):
        model.push(random.gauss(0, 0.0001))
    for _ in range(3):
        model.push(0.01)
    vote_right_after = model.vote()
    for _ in range(60):
        model.push(random.gauss(0, 0.0001))
    vote_later = model.vote()
    assert abs(vote_later) < abs(vote_right_after)


def test_vote_bounded():
    random.seed(2)
    model = HawkesMomentumModel(warmup=10)
    for _ in range(15):
        model.push(0.0001)
    for _ in range(10):
        model.push(0.05)
    assert -1.0 <= model.vote() <= 1.0
