from scipy.stats import norm

def generate_synthetic_stock_data(num_bars, x, delta=0.25, dt=0.1):
  result = []
  for _ in range(num_bars):
    result.append(x)
    x = x + norm.rvs(scale=delta ** 2 * dt)

  return result