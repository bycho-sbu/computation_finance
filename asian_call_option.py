import os
import math
import numpy as np

def price_asian_call_options(input_text: str | bytes) -> None:
    """
    Processes Asian call option parameters from a multi-line string or from a text file containing such data.

    This function expects each line to contain tab-separated values with the following parameters:
    risk_free_rate, time in years, number of steps, volatility, initial price, and strike price. It computes and prints
    the value (one per line) of the Asian call option for each line of input.

    Parameters:
    - input_data (str | bytes): A multi-line string with tab-separated values or a path to a text file containing such
                                data.

    Note:
    - The function directly prints (one output per line) the calculated option prices, hence no return value.
    """
    # decode bytes to text if needed
    if isinstance(input_text, bytes):
        try:
            text = input_text.decode("utf-8")
        except Exception:
            text = input_text.decode("latin-1")
    else:
        text = input_text

    # if a path is provided, read file content (reject json since not supported here)
    if isinstance(text, str) and os.path.exists(text):
        _, ext = os.path.splitext(text)
        if ext.lower() == ".json":
            raise ValueError(
                f"got a json file ({text}); expected a txt/csv/tsv file with six numeric fields per line: r, T, N, sigma, S0, K"
            )
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()

    # split into non-empty, non-comment lines
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and not ln.strip().startswith("#")]

    # mc controls (can be set via environment variables if desired)
    num_paths = int(os.environ.get("ASIAN_MC_PATHS", "100000"))
    seed = int(os.environ.get("ASIAN_MC_SEED", "12345"))
    antithetic = os.environ.get("ASIAN_MC_ANTITHETIC", "1") not in {"0", "false", "False"}

    for lineno, ln in enumerate(lines, start=1):
        parts = ln.split("\t") if "\t" in ln else ln.split(",")
        if len(parts) != 6:
            raise ValueError(f"line {lineno}: expected 6 values but got {len(parts)} in: {ln!r}")

        try:
            r, T, N, sigma, S0, K = parts
            r = float(r); T = float(T); N = int(float(N))
            sigma = float(sigma); S0 = float(S0); K = float(K)
        except Exception as e:
            raise ValueError(f"line {lineno}: failed to parse numbers {parts!r}: {e}") from e

        price = value_of_asian_call_option_mc(
            risk_free_rate=r,
            time=T,
            num_steps=N,
            volatility=sigma,
            initial_price=S0,
            strike_price=K,
            num_paths=num_paths,
            seed=seed,
            antithetic=antithetic,
        )
        print(f"{price:.6f}")

def value_of_asian_call_option_mc(
    risk_free_rate: float,
    time: float,
    num_steps: int,
    volatility: float,
    initial_price: float,
    strike_price: float,
    num_paths: int = 100000,
    seed: int = 12345,
    antithetic: bool = True,
) -> float:
    """
    estimates the price of an arithmetic-average asian call under gbm via monte carlo.

    payoff per path:
        a = (1/n) * sum_{i=1..n} s_{t_i},  t_i = i * dt
        payoff = max(a - k, 0)
    price = exp(-r * t) * e_q[payoff]

    notes:
    - uses log-euler discretization: s_{t+dt} = s_t * exp((r - 0.5*sigma^2) * dt + sigma*sqrt(dt)*z)
    - uses antithetic variates for variance reduction
    - iteratively updates s and running sum to avoid storing full path matrices
    """
    if time <= 0.0 or num_steps <= 0:
        return max(initial_price - strike_price, 0.0)
    if initial_price <= 0.0 or strike_price < 0.0 or num_paths <= 0:
        return 0.0

    dt = time / float(num_steps)

    # near-zero volatility deterministic path
    if volatility <= 1e-12:
        growth = math.exp(risk_free_rate * dt)
        if abs(growth - 1.0) < 1e-12:
            avg = initial_price
        else:
            series_sum = initial_price * (growth * (1.0 - growth**num_steps) / (1.0 - growth))
            avg = series_sum / float(num_steps)
        payoff = max(avg - strike_price, 0.0)
        return math.exp(-risk_free_rate * time) * payoff

    rng = np.random.default_rng(seed)

    if antithetic:
        m_half = (num_paths + 1) // 2
        Z_half = rng.standard_normal(size=(m_half, num_steps))
        Z = np.vstack([Z_half, -Z_half])[:num_paths, :]
    else:
        Z = rng.standard_normal(size=(num_paths, num_steps))

    drift = (risk_free_rate - 0.5 * (volatility ** 2)) * dt
    shock = volatility * math.sqrt(dt)

    S = np.full(shape=(num_paths,), fill_value=initial_price, dtype=float)
    sum_S = np.zeros(num_paths, dtype=float)

    for i in range(num_steps):
        np.multiply(S, np.exp(drift + shock * Z[:, i]), out=S)
        sum_S += S

    A = sum_S / float(num_steps)
    payoff = np.maximum(A - strike_price, 0.0)
    price = math.exp(-risk_free_rate * time) * float(np.mean(payoff))
    return price


if __name__ == "__main__":
    file_path = "D:\\Grad\\gradPrep\\workspace\\sample_data\\asian_call_price.tsv"
    price_asian_call_options(file_path)
