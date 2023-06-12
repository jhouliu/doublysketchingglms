sim.fn = function(param) {
  set.seed(param[["seed"]])
  n = param[["pop.size"]]
  p = param[["num.cov"]]
  X = switch(param[["X.distn"]],
             Uniform = runif(n * p, -1, 1),
             Normal = rnorm(n * p),
             Cauchy = rcauchy(n * p),
             t3 = rt(n * p, 3),
             t5 = rt(n * p, 5),
             t10 = rt(n * p, 10))
  X = X / p
  dim(X) = c(n, p)
  beta = rep(c(1, -1), length.out = p)
  eta = as.vector(X %*% beta)
  mu = switch(param[["link.fn"]],
              BinLogit = plogis(eta),
              BinCloglog = 1 - exp(-exp(eta)),
              PoiLog = exp(eta))
  y = switch(substr(param[["link.fn"]], 1, 3),
             Bin = rbinom(n, 1, mu),
             Poi = rpois(n, mu))
  
  fitfn = switch(param[["link.fn"]],
                 BinLogit = Double_Sketch_Binomial_Logit_full,
                 BinCloglog = Double_Sketch_Binomial_Cloglog_full,
                 PoiLog = Double_Sketch_Poisson_Log_full)

  glmfitfn = switch(param[["link.fn"]],
                    BinLogit = Logistic_IRLS,
                    BinCloglog = Cloglog_IRLS,
                    PoiLog = Poisson_IRLS)

  beta.init = rep(0, p)
  
  sketch.timing = system.time({
    sketch.fit <-
      fitfn(
        X,
        y,
        beta.init,
        m = param[["sketch1.size"]],
        k = param[["sketch2.size"]],
        iters = param[["iters"]]
      )
  })
  sketch.beta = sketch.fit$beta
  
  if (isTRUE(param[["fit.glm"]])) {
    glm.timing = system.time(glm.fit <- glmfitfn(X, y, beta.init, iters = 1000))
    glm.beta = tail(glm.fit$B, 1)[1,]
  } else {
    glm.timing = glm.beta = NULL
  }
  if (isFALSE(param[["return.data"]])) {
    X = NULL
    y = NULL
  }

  return(list(sketch.fit = sketch.fit, sketch.timing = sketch.timing,
              glm.fit = glm.fit, glm.timing = glm.timing,
              sketch.mse.beta0 = mean((sketch.beta - beta)^2),
              sketch.mse.betahat = mean((sketch.beta - glm.beta)^2),
              glm.mse.beta0 = mean((glm.beta - beta)^2), X = X, y = y
              ))
}

init.Poi.Log = function(X, y) {
  # https://hal.archives-ouvertes.fr/hal-01577698/document
  mu0 = y + 0.1
  eta0 = log(mu0)
  Z0 = eta0 + (y - mu0) / mu0
  W0 = mu0
  lm.wfit(X, Z0, W0)$coefficients
}

init.Bin.Logit = function(X, y) {
  mu0 = abs(y - 0.25)
  eta0 = qlogis(mu0)
  W0 = mu0 * (1 - mu0)
  Z0 = eta0 + (y - mu0) / W0
  lm.wfit(X, Z0, W0)$coefficients
}

init.Bin.cloglog = function(X, y) {
  mu0 = abs(y - 0.25)
  eta0 = log(-log(1 - mu0))
  Z0 = eta0 + (y - mu0) / ((mu0 - 1) * log(1 - mu0))
  W0 = (1 - mu0) * log(1 - mu0) ^ 2 / mu0
  lm(Z0 ~ X + 0, weights = W0)$coefficients
}
