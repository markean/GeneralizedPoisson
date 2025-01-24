"""Generalized Poisson Distribution."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes, validate_sample
from numpyro.util import is_prng_key


class GeneralizedPoisson(Distribution):
    """Generalized Poisson distribution.

    Args:
        lam (jax.Array): Rate parameter (lambda) of the distribution.
        delta (jax.Array): Dispersion parameter (delta) of the distribution.
        max_iters (int, optional): Maximum number of iterations for the
            rejection sampling. Defaults to `500`.
        validate_args (bool, optional): Whether to enable validation of
            distribution parameters and arguments to `.log_prob()` method.
            Defaults to `None`.
    """

    arg_constraints = {
        "lam": constraints.positive,
        "delta": constraints.interval(-1, 1),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        lam: jax.Array,
        delta: jax.Array,
        max_iters: int | None = 500,
        validate_args=None,
    ) -> None:
        """Initialize the GeneralizedPoisson."""
        self.lam, self.delta = promote_shapes(lam, delta)
        self.max_iters = max_iters
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.lam), jnp.shape(self.delta)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()) -> jax.Array:
        """Generate samples from the distribution.

        Args:
            key (jax.Array): A pseudo-random number generator key.
            sample_shape (tuple, optional): Batch size to be drawn from the
                distribution. Defaults to `()`.

        Returns:
            jax.Array: Samples from the distribution.
        """
        assert is_prng_key(key)

        return _generalized_poisson_rejection(
            key,
            self.lam,
            self.delta,
            self.max_iters,
            shape=sample_shape + self.batch_shape,
        )

    @validate_sample
    def log_prob(self, value) -> jax.Array:
        """Compute the log probability of a given value.

        Args:
            value (jax.Array): Value for which to compute the log probability.

        Returns:
            jax.Array: Log probability of the given value.
        """
        if self._validate_args:
            self._validate_sample(value)

        eps = jnp.finfo(float).eps
        # NOTE: The masking needs more investigation for numerical stability
        # mask = self.delta < jnp.maximum(-1.0, -self.lam / 4.0)
        mask = self.delta < jnp.maximum(-1.0, -self.lam / value)
        post_value = self.lam + self.delta * value
        safe_post_value = jnp.where(mask, eps, post_value)
        log_safe_post_value = jnp.where(mask, -10, jnp.log(safe_post_value))
        log_prob = (
            jnp.log(self.lam)
            + (value - 1) * log_safe_post_value
            - safe_post_value
            - jax.scipy.special.gammaln(value + 1)
        )

        return log_prob

    @property
    def mean(self):
        """Mean of the generalized Poisson distribution.

        Returns:
            jax.Array: Mean of the distribution.
        """
        return self.lam / (1.0 - self.delta)

    @property
    def variance(self):
        """Variance of the generalized Poisson distribution.

        Returns:
            jax.Array: Variance of the distribution.
        """
        return self.lam / lax.integer_pow(1.0 - self.delta, 3)


@partial(jit, static_argnames=("max_iters", "shape"))
def _generalized_poisson_rejection(key, lam, delta, max_iters, shape):
    lam = jnp.broadcast_to(lam, shape)
    lam = lax.convert_element_type(lam, jnp.float32)
    delta = jnp.broadcast_to(delta, shape)
    u = jax.random.uniform(key, shape)
    log_u = jnp.log(u)
    log_w = -delta
    log_s = -lam
    log_p = log_s
    x = jnp.zeros(shape, dtype=jnp.int32)
    accepted = log_u < log_s

    def cond_fun(carry):
        i, _, _, _, accepted = carry

        return (~accepted).any() & (i < max_iters)

    def body_fun(carry):
        i, x, log_s, log_p, accepted = carry

        x = x + jnp.where(~accepted, 1, 0)
        c = jnp.maximum(lam - delta + delta * x, jnp.finfo(float).eps)
        log_p += (
            log_w + jnp.log(c) + (x - 1) * jnp.log1p(delta / c) - jnp.log(x)
        )
        log_s = jnp.logaddexp(log_s, log_p)
        accepted = log_u < log_s
        i += 1

        return i, x, log_s, log_p, accepted

    return lax.while_loop(cond_fun, body_fun, (0, x, log_s, log_p, accepted))[
        1
    ].astype(jnp.int32)
