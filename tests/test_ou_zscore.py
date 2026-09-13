import math
import random

from models.ou_zscore import OUMeanReversionModel


def test_abstains_before_min_samples():
    model = OUMeanReversionModel(min_samples=60, refit_every=10)
    for _ in range(30):
        model.push(100.0)
    assert model.vote() == 0.0
    assert not model.is_fitted


def test_flags_reversion_on_synthetic_ou_series():
    random.seed(0)
    model = OUMeanReversionModel(window=500, min_samples=60, refit_every=20)
    # simulate a real OU process in log-price space, then push its exp()
    x, mu, theta, sigma = math.log(100.0), math.log(100.0), 0.08, 0.01
    for _ in range(300):
        x += theta * (mu - x) + random.gauss(0, sigma)
        model.push(math.exp(x))
    assert model.is_fitted
    # force the price sharply above its long-run mean and check the vote
    # leans toward reversion (negative -- expects it to fall back)
    model.push(math.exp(mu + 6 * sigma))
    vote = model.vote()
    assert vote < 0, f"expected a reversion (negative) vote, got {vote}"


def test_vote_bounded():
    random.seed(1)
    model = OUMeanReversionModel(window=500, min_samples=60, refit_every=20)
    x, mu, theta, sigma = math.log(50.0), math.log(50.0), 0.05, 0.02
    for _ in range(400):
        x += theta * (mu - x) + random.gauss(0, sigma)
        model.push(math.exp(x))
    model.push(math.exp(mu + 20 * sigma))  # an extreme deviation
    assert -1.0 <= model.vote() <= 1.0


def test_non_mean_reverting_window_does_not_crash():
    # a pure random walk (no reversion) -- OLS may find b close to or over
    # 1.0 in some windows; the model should just stay unfitted, not raise.
    random.seed(2)
    model = OUMeanReversionModel(min_samples=30, refit_every=10)
    x = math.log(10.0)
    for _ in range(100):
        x += random.gauss(0, 0.01)
        model.push(math.exp(x))
    assert model.vote() in (0.0, model.vote())  # just asserting no exception
