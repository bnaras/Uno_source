## uno_version reports the linked Uno version
expect_true(grepl("^[0-9]+\\.[0-9]+\\.[0-9]+$", uno_version()))

## uno_solve solves a simple convex NLP (filtersqp + HiGHS)
## min (x1-1)^2 + (x2-2)^2  s.t.  x1 + x2 <= 10
## the unconstrained optimum (1, 2) is feasible, so x* = (1, 2), f* = 0.
obj  <- function(x) (x[1] - 1)^2 + (x[2] - 2)^2
grad <- function(x) c(2 * (x[1] - 1), 2 * (x[2] - 2))
cons <- function(x) x[1] + x[2]
jac  <- function(x) c(1, 1)                              # rows{0,0} cols{0,1}
hess <- function(x, sigma, lambda) c(2 * sigma, 2 * sigma)  # rows{0,1} cols{0,1}

res <- uno_solve(
  n = 2L, lb = c(-Inf, -Inf), ub = c(Inf, Inf), sense = "minimize",
  obj = obj, grad = grad,
  m = 1L, cl = -Inf, cu = 10, cons = cons,
  jac_rows = c(0L, 0L), jac_cols = c(0L, 1L), jac = jac,
  hess_rows = c(0L, 1L), hess_cols = c(0L, 1L), hess = hess,
  x0 = c(0, 0), preset = "filtersqp", base_indexing = 0L, verbose = FALSE
)
expect_equal(res$optimization_status, 0L)            # UNO_SUCCESS
expect_equal(res$solution_status, 1L)                 # UNO_FEASIBLE_KKT_POINT
expect_equal(res$objective, 0, tolerance = 1e-6)
expect_equal(res$primal, c(1, 2), tolerance = 1e-5)

## an R error inside a callback aborts cleanly via UNO_EVALUATION_ERROR
## (R_tryEval must catch the error rather than longjmp through Uno's C++ stack)
obj_err <- function(x) stop("boom")
res_err <- uno_solve(
  n = 2L, lb = c(-Inf, -Inf), ub = c(Inf, Inf), sense = "minimize",
  obj = obj_err, grad = grad,
  m = 1L, cl = -Inf, cu = 10, cons = cons,
  jac_rows = c(0L, 0L), jac_cols = c(0L, 1L), jac = jac,
  hess_rows = c(0L, 1L), hess_cols = c(0L, 1L), hess = hess,
  x0 = c(0, 0), preset = "filtersqp", base_indexing = 0L, verbose = FALSE
)
expect_equal(res_err$optimization_status, 3L)         # UNO_EVALUATION_ERROR

## the interior-point "ipopt" preset uses MUMPS as its symmetric-indefinite
## linear solver. MUMPS is reached at runtime from the 'rmumps' package via
## R_FindSymbol (see src/dmumps_shim.c) -- this exercises that path end-to-end.
## Same convex NLP: x* = (1, 2), f* = 0.
res_ip <- uno_solve(
  n = 2L, lb = c(-Inf, -Inf), ub = c(Inf, Inf), sense = "minimize",
  obj = obj, grad = grad,
  m = 1L, cl = -Inf, cu = 10, cons = cons,
  jac_rows = c(0L, 0L), jac_cols = c(0L, 1L), jac = jac,
  hess_rows = c(0L, 1L), hess_cols = c(0L, 1L), hess = hess,
  x0 = c(0, 0), preset = "ipopt", base_indexing = 0L, verbose = FALSE
)
expect_equal(res_ip$optimization_status, 0L)          # UNO_SUCCESS
expect_equal(res_ip$solution_status, 1L)              # UNO_FEASIBLE_KKT_POINT
expect_equal(res_ip$objective, 0, tolerance = 1e-6)
expect_equal(res_ip$primal, c(1, 2), tolerance = 1e-5)
