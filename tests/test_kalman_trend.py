import math
import random

from models.kalman_trend import KalmanTrendModel


def test_abstains_during_warmup():
    model = KalmanTrendModel(warmup=20)
    for _ in range(10):
        model.push(100.0)
    assert model.vote() == 0.0


def test_flags_upward_trend():
    random.seed(0)
    model = KalmanTrendModel(warmup=20)
    price = 100.0
    for _ in range(200):
        price *= math.exp(0.0005 + random.gauss(0, 0.0003))
        model.push(price)
    vote = model.vote()
    assert vote > 0.3, f"expected an upward-trend vote, got {vote}"


def test_flags_downward_trend():
    random.seed(1)
    model = KalmanTrendModel(warmup=20)
    price = 100.0
    for _ in range(200):
        price *= math.exp(-0.0005 + random.gauss(0, 0.0003))
        model.push(price)
    vote = model.vote()
    assert vote < -0.3, f"expected a downward-trend vote, got {vote}"


def test_flat_series_votes_near_zero():
    random.seed(2)
    model = KalmanTrendModel(warmup=20)
    price = 100.0
    for _ in range(200):
        price *= math.exp(random.gauss(0, 0.0003))
        model.push(price)
    assert abs(model.vote()) < 0.3


def test_ignores_non_positive_prices():
    model = KalmanTrendModel(warmup=5)
    model.push(100.0)
    model.push(0.0)
    model.push(-5.0)
    model.push(101.0)
    # should not raise, and should still be tracking normally
    assert model.sample_size >= 2
