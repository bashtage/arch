# Reference values for the VaR backtests in arch.univariate.backtest.
#
# The constants REFERENCE, BASEL_TABLE2_CDF, BINOM_ZONES and KUPIEC_REGIONS at the top
# of arch/tests/univariate/test_backtest.py were produced by this script with R 4.6.1
# and rugarch 1.5.6 (the long vectors were rewrapped to fit the line length).  The
# script prints them to stdout in Python syntax, followed by the probabilities of the
# Basel (1996) Tables 1 and 2 as comments.  Run it from any directory with
#
#   Rscript var_backtest_reference.R > var_backtest_reference.txt
#
# It needs rugarch (install.packages("rugarch")); set R_LIBS_USER to use a private
# library.  The R package GAS (BacktestVaR) is no longer on CRAN, so it is not used.
#
# rugarch counts an exception as actual < VaR with VaR a negative quantile, so the
# arch input `var` (a positive loss) is passed as -var.  Its LR_cc is LR_uc + LR_ind,
# with LR_ind computed on the T-1 transitions (the first observation is conditioned
# on), the same as arch.
suppressMessages(library(rugarch))
cat("# rugarch", as.character(packageVersion("rugarch")), R.version.string, "\n\n")

# ---- an independent implementation: log-likelihoods, with 0 log 0 = 0 -------------
ll <- function(k, n, q) {              # k successes in n Bernoulli(q) trials
    (if (k > 0) k * log(q) else 0) + (if (n - k > 0) (n - k) * log1p(-q) else 0)
}
mle <- function(k, n) {                # numerical ML, checked against k/n
    if (n == 0) return(0)
    if (k == 0) return(0)
    if (k == n) return(1)
    o <- optimize(function(q) ll(k, n, q), c(1e-12, 1 - 1e-12), maximum = TRUE, tol = 1e-12)
    stopifnot(abs(o$maximum - k / n) < 1e-6)
    k / n
}
lr_uc <- function(I, p) {
    n <- length(I); k <- sum(I)
    2 * (ll(k, n, mle(k, n)) - ll(k, n, p))
}
lr_ind <- function(I) {
    a <- head(I, -1); b <- tail(I, -1)
    n00 <- sum(a == 0 & b == 0); n01 <- sum(a == 0 & b == 1)
    n10 <- sum(a == 1 & b == 0); n11 <- sum(a == 1 & b == 1)
    ll1 <- ll(n01, n00 + n01, mle(n01, n00 + n01)) + ll(n11, n10 + n11, mle(n11, n10 + n11))
    n <- n00 + n01 + n10 + n11; k <- n01 + n11
    ll0 <- ll(k, n, mle(k, n))
    2 * (ll1 - ll0)
}
pv <- function(stat, df) pchisq(stat, df, lower.tail = FALSE)
fmt <- function(x) formatC(x, digits = 17, format = "g")

pyvec <- function(v) paste0("[", paste(v, collapse = ", "), "]")
bits <- function(I) paste(I, collapse = "")

# one case: prints a python dict entry. `rug` is TRUE when rugarch can be called
emit <- function(name, I, p, returns = NULL, var = NULL, rug = TRUE, note = "") {
    uc <- lr_uc(I, p); ind <- lr_ind(I); cc <- uc + ind
    cat(sprintf("    # %s\n", note))
    cat(sprintf("    \"%s\": {\n", name))
    if (is.null(returns)) {
        cat(sprintf("        \"pattern\": \"%s\",\n", bits(I)))
    } else {
        cat(sprintf("        \"returns\": %s,\n", pyvec(returns)))
        cat(sprintf("        \"var\": %s,\n", pyvec(var)))
    }
    cat(sprintf("        \"p\": %s,\n", p))
    cat(sprintf("        \"nexc\": %d,\n", sum(I)))
    cat(sprintf("        \"uc\": (%s, %s),\n", fmt(uc), fmt(pv(uc, 1))))
    cat(sprintf("        \"ind\": (%s, %s),\n", fmt(ind), fmt(pv(ind, 1))))
    cat(sprintf("        \"cc\": (%s, %s),\n", fmt(cc), fmt(pv(cc, 2))))
    if (rug) {
        act <- if (is.null(returns)) ifelse(I == 1, -1.5, 0.5) else returns
        vr <- if (is.null(var)) rep(-1.0, length(I)) else -var
        r <- VaRTest(alpha = p, actual = act, VaR = vr)
        cat(sprintf("        \"rugarch_uc\": (%s, %s),\n", fmt(r$uc.LRstat), fmt(r$uc.LRp)))
        cat(sprintf("        \"rugarch_cc\": (%s, %s),\n", fmt(r$cc.LRstat), fmt(r$cc.LRp)))
        stopifnot(abs(r$uc.LRstat - uc) < 1e-9, abs(r$cc.LRstat - cc) < 1e-9)
        stopifnot(abs(r$uc.LRp - pv(uc, 1)) < 1e-12, abs(r$cc.LRp - pv(cc, 2)) < 1e-12)
        # rugarch has no separate independence test; recover it as cc - uc
        cat(sprintf("        \"rugarch_ind\": %s,\n", fmt(r$cc.LRstat - r$uc.LRstat)))
        stopifnot(abs((r$cc.LRstat - r$uc.LRstat) - ind) < 1e-9)
    }
    cat("    },\n")
}

cat("REFERENCE = {\n")

# ---- series built from returns and VaR (pandas/array path) -------------------------
mk_returns <- function(seed, n, p, scale, df) {
    set.seed(seed)
    sigma <- round(runif(n, 0.8, 1.6), 4)
    r <- round(sigma * scale * rt(n, df), 4)
    v <- round(sigma * qnorm(1 - p), 4)
    list(r = r, v = v, I = as.integer(-r > v))
}
for (spec in list(list(11, 100, 0.05, 0.9, 5), list(12, 100, 0.05, 1.1, 4),
                  list(13, 120, 0.05, 0.7, 6))) {
    d <- do.call(mk_returns, spec)
    emit(sprintf("returns_seed%d", spec[[1]]), d$I, spec[[3]], d$r, d$v,
         note = sprintf("set.seed(%d); n=%d, p=%g, t(%d) returns, time-varying VaR",
                        spec[[1]], spec[[2]], spec[[3]], spec[[5]]))
}

# ---- 0/1 exception patterns --------------------------------------------------------
set.seed(1); I1 <- rbinom(500, 1, 0.01)
emit("iid_p01_n500", I1, 0.01, note = "set.seed(1); rbinom(500, 1, 0.01)")
set.seed(2); I2 <- rbinom(250, 1, 0.05)
emit("iid_p05_n250", I2, 0.05, note = "set.seed(2); rbinom(250, 1, 0.05)")
set.seed(3)                                  # first-order Markov chain, clustered
I3 <- integer(400); I3[1] <- 0
for (t in 2:400) I3[t] <- rbinom(1, 1, if (I3[t - 1] == 1) 0.4 else 0.03)
emit("clustered_p05_n400", I3, 0.05, note = "set.seed(3); Markov chain p01=0.03, p11=0.4, n=400")
I4 <- as.integer(seq_len(140) %% 7 == 0)     # isolated exceptions: no 1 -> 1 transition
emit("no_11_transitions", I4, 0.1, note = "an exception every 7th day, n=140: n11 = 0")
I5 <- as.integer(seq_len(60) %% 2 == 1)      # alternating 1,0,1,0: n00 = n11 = 0
emit("alternating", I5, 0.5, note = "alternating 1,0,...: n00 = n11 = 0")
I6 <- c(rep(0L, 30), rep(1L, 6), rep(0L, 64))  # a single cluster
emit("single_cluster", I6, 0.05, note = "one run of 6 exceptions in n=100")
I7 <- c(0L, rep(1L, 4), rep(0L, 15), 1L, 1L, rep(0L, 28))
emit("two_clusters_short", I7, 0.1, note = "n=50, clusters of 4 and 2")
I8 <- c(rep(0L, 12), 1L, rep(0L, 12))        # a single isolated exception
emit("single_exception", I8, 0.04, note = "n=25 with one exception: n11 = 0, n10 = 1")

# edge cases that rugarch cannot compute (it indexes a 2x2 table) are computed by the
# independent implementation only; the Kupiec statistic is also compared with
# rugarch:::.LR.uc, which handles k=0 and k=n through R's 0^0 = 1.
edge <- list(zero_exceptions = list(I = rep(0L, 250), p = 0.01),
             all_exceptions = list(I = rep(1L, 40), p = 0.05),
             first_only = list(I = c(1L, rep(0L, 49)), p = 0.05),
             last_only = list(I = c(rep(0L, 49), 1L), p = 0.05),
             two_obs = list(I = c(0L, 1L), p = 0.5),
             two_obs_11 = list(I = c(1L, 1L), p = 0.3))
for (nm in names(edge)) {
    e <- edge[[nm]]
    emit(nm, e$I, e$p, rug = FALSE, note = sprintf("edge case, independent implementation (%s)", nm))
    uc_r <- rugarch:::.LR.uc(p = e$p, TN = length(e$I), N = sum(e$I))
    stopifnot(abs(uc_r - lr_uc(e$I, e$p)) < 1e-9)
}
cat("}\n\n")

# ---- Kupiec and traffic light: binomial references ---------------------------------
cat("BASEL_TABLE2_CDF = ", pyvec(sapply(0:15, function(x) fmt(pbinom(x, 250, 0.01)))), "\n", sep = "")
cat("BINOM_ZONES = [\n")
for (s in list(c(250, 0.01), c(250, 0.05), c(500, 0.01), c(1000, 0.01), c(100, 0.05),
               c(60, 0.05), c(250, 0.001), c(20, 0.001), c(20, 0.5))) {
    n <- s[1]; p <- s[2]
    cat(sprintf("    # n=%g, p=%g: qbinom(0.95), qbinom(0.9999)\n", n, p))
    cat(sprintf("    (%g, %g, %d, %d),\n", n, p, qbinom(0.95, n, p), qbinom(0.9999, n, p)))
}
cat("]\n\n")

# Kupiec (1995) non-rejection regions at the 5% test size: the exceptions N for which
# the LR_uc is below the chi2(1) 95% critical value
cat("KUPIEC_REGIONS = {\n")
for (p in c(0.01)) for (T in c(255, 510, 1000)) {
    ok <- sapply(0:T, function(N) lr_uc(c(rep(1L, N), rep(0L, T - N)), p) <= qchisq(0.95, 1))
    Ns <- which(ok) - 1
    cat(sprintf("    (%g, %g): (%d, %d),\n", p, T, min(Ns), max(Ns)))
}
cat("}\n\n")

# The Basel numbers, rounded as printed in the BCBS (1996) Table 2
cat("# Basel table 2 cdf (%):", round(100 * pbinom(0:10, 250, 0.01), 2), "\n")
cat("# Table 1 exact 99% (%):", round(100 * dbinom(0:15, 250, 0.01), 1), "\n")
cat("# Table 1 type 1 99% (%):", round(100 * pbinom(-1:14, 250, 0.01, lower.tail = FALSE), 1), "\n")
