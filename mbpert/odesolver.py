import torch

# These constants are definied the same as in `scipy.integrate.solve_ivp`
SAFETY = 0.9
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class RK45():
  """explicit Runge-Kutta method of order 5(4)"""

  # Butcher tableau 
  # https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/integrate/_ivp/rk.py#L280
  n_stages = 6
  error_estimator_order = 4
  C = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1])
  A = torch.tensor([
      [0, 0, 0, 0, 0],
      [1/5, 0, 0, 0, 0],
      [3/40, 9/40, 0, 0, 0],
      [44/45, -56/15, 32/9, 0, 0],
      [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
      [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
  ])
  B = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
  E = torch.tensor([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])

  def __init__(self, fun, t_span, args, first_step=1e-3):
    # Begining and end time points of integration
    self.t0, self.tf = map(float, t_span)
    
    # This line is from the source code of scipy's `solve_ivp`. It wraps the user
    # function in a lambda to hide additional arguments.
    self.fun = lambda t, x, fun=fun: fun(t, x, *args)

    # Relative and absolute tolerance as in solve_ivp
    self.rtol, self.atol = 1e-3, 1e-6

    # Initial step size, assume direction of integration is always positive, i.e, h > 0
    self.h0 = first_step

    # For use in updating step size
    # https://en.wikipedia.org/wiki/Adaptive_step_size#Embedded_error_estimates
    self.error_exponent = -1 / (self.error_estimator_order + 1)


  # This is a rewrite of `rk_step` in scipy.integrate.solve_ivp source code using Pytorch tensors
  def _step_rk(self, t, y, f, h, K):
    """Perform a single Runge-Kutta step.
    Parameters as in https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/integrate/_ivp/rk.py#L14

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.
    """
    K[0] = f
    for s, (a, c) in enumerate(zip(RK45.A[1:], RK45.C[1:]), start=1):
        dy = torch.matmul(torch.t(K[:s]), a[:s]) * h
        K[s] = self.fun(t + c * h, y + dy)

    y_new = y + h * torch.matmul(torch.t(K[:-1]), RK45.B)
    f_new = self.fun(t + h, y_new)

    K[-1] = f_new

    return y_new, f_new    

  # These two methods are equivalent to those in scipy `solve_ivp`
  def _estimate_error(self, K, h):
    return torch.matmul(torch.t(K), self.E) * h

  # NOTE: the `.item()` following `norm()` is essential, omitting it will get
  # runtime error "inplace modification of variables requiring gradient" that
  # is hard to debug!
  def _estimate_error_norm(self, K, h, scale):
    x = self._estimate_error(K, h) / scale
    return torch.linalg.norm(x).item() / x.numel() ** 0.5 
    

  # This is a rewrite of `_step_impl` using Pytorch tensors
  def _step_impl(self):
    # Current time, state and derivative 
    t, y, f = self.t, self.y, self.f

    rtol = self.rtol
    atol = self.atol
    h = self.h # current step size

    step_accepted = False
    step_rejected = False

    while not step_accepted:
      t_new = t + h

      if (t_new - self.tf) > 0:
        t_new = self.tf

      h = t_new - t

      y_new, f_new = self._step_rk(t, y, f, h, self.K)
      scale = atol + torch.maximum(torch.abs(y), torch.abs(y_new)) * rtol
      error_norm = self._estimate_error_norm(self.K, h, scale)

      if error_norm < 1:
          if error_norm == 0:
              factor = MAX_FACTOR
          else:
              factor = min(MAX_FACTOR, SAFETY * error_norm ** self.error_exponent)

          if step_rejected:
              factor = min(1, factor)

          h *= factor

          step_accepted = True
      else:
          h *= max(MIN_FACTOR, SAFETY * error_norm ** self.error_exponent)
          step_rejected = True

    # Update time, state, derivative and step size
    self.t, self.y, self.f, self.h = t_new, y_new, f_new, h


  def solve(self, y):
    # Initial time, step size, state and derivative dy/dt = f(t,y)
    self.t = self.t0
    self.h = self.h0
    self.y = y
    self.f = self.fun(self.t0, self.y)

    # Storage for RK stages
    self.K = torch.empty((RK45.n_stages + 1, y.numel()), dtype=y.dtype)
    
    while self.t < self.tf:
      self._step_impl()

    # if self.f.abs().mean().item() > 1e-3:
      # print(f'Mean absolute derivative at t={self.t}: {self.f.abs().mean().item()}')
      # print("ODE solver did not converge to steady state solution in the given time range.")

    return self.y