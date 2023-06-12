// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; indent-tabs-mode: nil; -*-
#define ARMA_WARN_LEVEL 1
#define ARMA_NO_DEBUG
// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

//' Fit logistic regression using IRLS in C++
//' 
//' @param X A numeric matrix with the desired covariates. If an intercept is 
//' desired, then a column of ones should be part of this input matrix.
//' @param y A vector of responses taking values zero or one. Should be the same
//' length as the number of rows of X.
//' @param beta The initial value of the coefficients.
//' @param iters The maximum number of IRLS iterations to run.
//' @param eps The convergence threshold for the L1 norm between successive
//' iterations of the parameters during the IRLS iteration.
//' 
//' @details
//' This function skips a lot of sanity checks and input validation that the 
//' usual R functions do in the name of speed. There are also no guardrails 
//' against unstable numerical computations.
//' 
//' @returns
//' Returns a list with two elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration and `cov` is the 
//' Hessian matrix at the final iteration.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Logistic_IRLS(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int iters = 10, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec mu(n), diff(p);

  for (unsigned int iter = 0; iter < iters; iter++) {
    mu = (arma::tanh(X * beta / 2) + 1)/2;

    hess_mat = X.t() * (X.each_col() % (mu % (1 - mu)));
    diff = arma::solve(hess_mat, X.t() * (y - mu), arma::solve_opts::likely_sympd);
    if (arma::accu(arma::abs(diff)) < eps) {
      B = B.head_rows(iter);
      break;
    }
    beta += diff;
    B.row(iter) = beta.t();
  }
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat
  );
}

// [[Rcpp::export]]
arma::vec Logistic_Iter(const arma::mat& Xs, const arma::vec& ys, const arma::vec& beta) {
  // arma::vec mu = 1 / (1 + arma::exp(-Xs * beta));
  arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
  return arma::solve(Xs.t() * (Xs.each_col() % (mu % (1 - mu))),
                     Xs.t() * (ys - mu),
                     arma::solve_opts::likely_sympd);
}

//' Fit Binomial-Cloglog GLM using IRLS in C++
//' 
//' @param X A numeric matrix with the desired covariates. If an intercept is 
//' desired, then a column of ones should be part of this input matrix.
//' @param y A vector of responses taking values zero or one. Should be the same
//' length as the number of rows of X.
//' @param beta The initial value of the coefficients.
//' @param iters The maximum number of IRLS iterations to run.
//' @param eps The convergence threshold for the L1 norm between successive
//' iterations of the parameters during the IRLS iteration.
//' 
//' @details
//' This function skips a lot of sanity checks and input validation that the 
//' usual R functions do in the name of speed. Means are clamped to be between
//' approximately 2.2e-16 and 1 - 2.2e-16.
//' 
//' @returns
//' Returns a list with two elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration and `cov` is the 
//' Hessian matrix at the final iteration.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Cloglog_IRLS(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int iters = 10, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec eta(n), mu(n), w(n), z(n), diff(p);
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    eta = X * beta;
    mu = -arma::expm1(-arma::exp(eta));
    mu = arma::clamp(mu, 2.220446e-16, 1 - 2.220446e-16);
    z = eta + (y - mu) / ((mu - 1) % arma::log1p(-mu));
    w = (1 - mu) / mu % arma::square(arma::log1p(-mu));
    
    hess_mat = X.t() * (X.each_col() % w);
    diff = arma::solve(hess_mat, X.t() * (z % w), arma::solve_opts::likely_sympd);
    if (arma::accu(arma::abs(beta - diff)) < eps) {
      B = B.head_rows(iter);
      break;
    }
    beta = diff;
    B.row(iter) = beta.t();
  }
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat
  );
}

//' Fit Poisson-Log regression using IRLS in C++
//' 
//' @param X A numeric matrix with the desired covariates. If an intercept is 
//' desired, then a column of ones should be part of this input matrix.
//' @param y A vector of responses taking values zero or one. Should be the same
//' length as the number of rows of X.
//' @param beta The initial value of the coefficients.
//' @param iters The maximum number of IRLS iterations to run.
//' @param eps The convergence threshold for the L1 norm between successive
//' iterations of the parameters during the IRLS iteration.
//' 
//' @details
//' This function skips a lot of sanity checks and input validation that the 
//' usual R functions do in the name of speed. There are also no guardrails 
//' against unstable numerical computations.
//' 
//' @returns
//' Returns a list with two elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration and `cov` is the 
//' Hessian matrix at the final iteration.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Poisson_IRLS(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int iters = 10, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec mu(n), diff(p);
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    mu = arma::exp(X * beta);
    
    hess_mat = X.t() * (X.each_col() % mu);
    diff = arma::solve(hess_mat, X.t() * (y - mu),
                       arma::solve_opts::likely_sympd);
    if (arma::accu(arma::abs(diff)) < eps) {
      B = B.head_rows(iter);
      break;
    }
    beta += diff;
    B.row(iter) = beta.t();
  }
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat
  );
}

//' Fit Gamma-Inverse regression using IRLS in C++
//' 
//' @param X A numeric matrix with the desired covariates. If an intercept is 
//' desired, then a column of ones should be part of this input matrix.
//' @param y A vector of responses taking values zero or one. Should be the same
//' length as the number of rows of X.
//' @param beta The initial value of the coefficients.
//' @param iters The maximum number of IRLS iterations to run.
//' @param eps The convergence threshold for the L1 norm between successive
//' iterations of the parameters during the IRLS iteration.
//' @param backtrack An integer determining the number of times to backtrack an
//' invalid parameter update. Each backtracking step consists of halving the 
//' update difference, which is as done by [stats::glm.fit] internally.
//' 
//' @returns
//' Returns a list with three elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration, `cov` is the 
//' Hessian matrix at the final iteration, and `dispersion` which is the fitted
//' dispersion value.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Gamma_Inverse_IRLS(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int iters = 10, const unsigned int backtrack = 100, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec eta(n), diff(p);
  eta = X * beta;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    arma::mat Xw = X.each_col() / eta;
    arma::vec zw = 2 - eta % y;
    
    hess_mat = Xw.t() * Xw;
    diff = arma::solve(hess_mat, Xw.t() * zw, arma::solve_opts::likely_sympd);
    
    for (unsigned int inner_iter = 0; inner_iter < backtrack; inner_iter++) {
      eta = X * diff;
      if (eta.is_finite() && arma::all(eta > 0)) {break;}
      diff = (beta + diff) / 2;
    }
    double delta = arma::accu(arma::abs(beta - diff));
    beta = diff;
    B.row(iter) = beta.t();
    if (delta < eps) {B = B.head_rows(iter + 1); break;}
  }
  
  double dispersion = arma::accu(arma::square(y % eta - 1)) / (n - p);
  
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat,
    Rcpp::_["dispersion"] = dispersion
  );
}

//' Fit a weighted Gamma-Inverse regression using IRLS in C++
//' 
//' @description
//' This is a helper function to perform weighted Gamma-Inverse GLM fit for the
//' Optimal Distributed Subsampling for Maximum Quasi-Likelihood Estimators With 
//' Massive Data method by Yu et al. (2022). This is so that the code is similar
//' and so the speeds are comparable.
//' 
//' @param X,y,beta,iters,backtrack,eps As in [Gamma_Inverse_IRLS].
//' @param w A vector of weights.
//' 
//' @returns
//' Returns a list with three elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration, `cov` is the 
//' Hessian matrix at the final iteration, and `dispersion` which is the fitted
//' dispersion value.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Gamma_Inverse_wIRLS(
    const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta,
    const unsigned int iters = 10, const unsigned int backtrack = 100, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec eta(n), diff(p);
  eta = X * beta;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    arma::mat Xw = X.each_col() % (arma::sqrt(w) / eta);
    arma::vec zw = (2 - eta % y) % arma::sqrt(w);
    
    hess_mat = Xw.t() * Xw;
    diff = arma::solve(hess_mat, Xw.t() * zw, arma::solve_opts::likely_sympd);
    
    for (unsigned int inner_iter = 0; inner_iter < backtrack; inner_iter++) {
      eta = X * diff;
      if (eta.is_finite() && arma::all(eta > 0)) {break;}
      diff = (beta + diff) / 2;
    }
    double delta = arma::accu(arma::abs(beta - diff));
    beta = diff;
    B.row(iter) = beta.t();
    if (delta < eps) {B = B.head_rows(iter + 1); break;}
  }
  
  double dispersion = arma::accu(arma::square(y % eta - 1)) / (n - p);
  
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat,
    Rcpp::_["dispersion"] = dispersion
  );
}

//' Fit a weighted logistic regression using IRLS in C++
//' 
//' @description
//' This is a helper function to perform weighted logistic GLM fit for the
//' Optimal Distributed Subsampling for Maximum Quasi-Likelihood Estimators With 
//' Massive Data method by Yu et al. (2022). This is so that the code is similar
//' and so the speeds are comparable.
//' 
//' @param X,y,beta,iters,eps As in [Logistic_IRLS].
//' @param w A vector of weights.
//' 
//' @returns
//' Returns a list with two elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration and `cov` is the 
//' Hessian matrix at the final iteration.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Logistic_wIRLS(
    const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta,
    const unsigned int iters = 10, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec mu(n), diff(p);
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    mu = (arma::tanh(X * beta / 2) + 1)/2;
    
    hess_mat = X.t() * (X.each_col() % (w % mu % (1 - mu)));
    diff = arma::solve(hess_mat, X.t() * (w % (y - mu)), arma::solve_opts::likely_sympd);
    if (arma::accu(arma::abs(diff)) < eps) {
      B = B.head_rows(iter);
      break;
    }
    beta += diff;
    B.row(iter) = beta.t();
  }
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat
  );
}

//' Fit a weighted Poisson-Log regression using IRLS in C++
//' 
//' @description
//' This is a helper function to perform weighted Poisson-Log GLM fit for the
//' Optimal Distributed Subsampling for Maximum Quasi-Likelihood Estimators With 
//' Massive Data method by Yu et al. (2022). This is so that the code is similar
//' and so the speeds are comparable.
//' 
//' @param X,y,beta,iters,eps As in [Logistic_IRLS].
//' @param w A vector of weights.
//' 
//' @returns
//' Returns a list with two elements: `B` which is a matrix where each row is 
//' the parameter at the corresponding IRLS iteration and `cov` is the 
//' Hessian matrix at the final iteration.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Poisson_wIRLS(
    const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta,
    const unsigned int iters = 10, const double eps = 1e-10
) {
  unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::value(arma::datum::nan)), hess_mat(p, p);
  arma::vec mu(n), diff(p);
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    mu = arma::exp(X * beta);
    
    hess_mat = X.t() * (X.each_col() % (w % mu));
    diff = arma::solve(hess_mat, X.t() * (w % (y - mu)),
                       arma::solve_opts::likely_sympd);
    if (arma::accu(arma::abs(diff)) < eps) {
      B = B.head_rows(iter);
      break;
    }
    beta += diff;
    B.row(iter) = beta.t();
  }
  // return B;
  return Rcpp::List::create(
    Rcpp::_["B"] = B,
    Rcpp::_["cov"] = hess_mat
  );
}
