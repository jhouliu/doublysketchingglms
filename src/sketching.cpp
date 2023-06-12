// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; indent-tabs-mode: nil; -*-
// #define ARMA_WARN_LEVEL 1
// #define ARMA_NO_DEBUG
// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

// [[Rcpp::plugins(cpp17)]]

//' Perform doubly-sketching on a logistic regression based on data in memory
//' using C++.
//' 
//' @param X A numeric matrix with the desired covariates. If an intercept is 
//' desired, then a column of ones should be part of this input matrix.
//' @param y A vector of responses taking values zero or one. Should be the same
//' length as the number of rows of X.
//' @param beta The initial value of the coefficients.
//' @param m The size of the first uniform sketch.
//' @param k The size of the second Clarkson-Woodruff sketch. If `k` is larger 
//' than `m`, then the second sketch is omitted.
//' @param iters The number of iterations to run for.
//' @param gamma The regularization constant for the Hessian matrices.
//' @param reset_iter The iteration at which to reset the determinantally 
//' averaged inverse Hessian matrices. Should be set to either 1 or Inf (to 
//' disable resetting).
//' @param reset_length The starting number of iterations between resets. Every
//' reset will increment this length by one internally. Should be set to 1 
//' unless there is a very unusual reason to change this.
//' 
//' @returns
//' Returns a list with two elements: `beta` is the final estimated parameter 
//' estimate, `B` which is a matrix tracing the parameter from iteration to 
//' itertation, and a matrix `Fhat_m_num` and scalar `Fhat_m_den`. Together, 
//' these two latter elements are the determinantal average. The estimate for 
//' the covariance matrix estimate can be obtained by using the expression
//' `Fhat_m_num / Fhat_m_den / m`. The `m` divisor is required as the internal
//' calculation avoids dividing and multiplying by `m` unnecessarily.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List Double_Sketch_Binomial_Logit(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  arma::mat B(iters, p, arma::fill::zeros);
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros), Xssw(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        arma::mat Xs = Xx.rows(csum, csum + rbinom - 1);
        arma::vec ys = yy.subvec(csum, csum + rbinom - 1);
        arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
        arma::vec w = arma::sqrt(mu % (1 - mu));
        w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
        arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
        zss(j) = arma::accu(flip % (ys - mu));
        Xss.row(j) = flip.t() * Xs;
        Xssw.row(j) = (flip % w).t() * Xs;
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
      arma::vec w = mu % (1 - mu);
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      Xss = Xs;
      Xssw = Xs.each_col() % arma::sqrt(w);
      zss = (ys - mu);
    }
    
    arma::mat hess_mat = Xssw.t() * Xssw;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    B.row(iter) = beta.t();
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["B"] = B,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Binomial_Logit_fold(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10L, const double gamma = 1,
    unsigned int reset_iter = 1L, unsigned int reset_length = 1L
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  arma::mat B(iters, p, arma::fill::zeros);
  
  for (unsigned int iter = 0L; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros), Xssw(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows), 
        mu = (arma::tanh(Xx * beta / 2) + 1)/2,
        w = arma::sqrt(mu % (1 - mu)),
        flip = arma::randi<arma::vec>(m, arma::distr_param(0, 1)) * 2L - 1L;
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      Xx.each_col() %= flip;
      yy = flip % (yy - mu); // yy now stores (yy - mu) for a single multiplication by flip
      
      unsigned int csum = 0L;
      for (unsigned int j = 0L; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0L) continue;
        zss(j) = arma::accu(yy.subvec(csum, csum + rbinom - 1L));
        Xss.row(j) = arma::sum(Xx.rows(csum, csum + rbinom - 1L), 0);
        Xssw.row(j) = w.subvec(csum, csum + rbinom - 1).t() * Xx.rows(csum, csum + rbinom - 1L);
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
      arma::vec w = mu % (1 - mu);
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      Xss = Xs;
      Xssw = Xs.each_col() % arma::sqrt(w);
      zss = (ys - mu);
    }
    
    arma::mat hess_mat = Xssw.t() * Xssw;
    hess_mat.diag() += gamma / (sqrt(1L + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1L);
    B.row(iter) = beta.t();
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["B"] = B,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List CW_Iter_Binomial_Logit(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    arma::mat Fhat_m_num, double Fhat_m_den,
    const unsigned int k, const double gamma = 1, unsigned int iter = 1
) {
  const unsigned int m = X.n_rows, p = X.n_cols;
  arma::mat Xss(k, p, arma::fill::zeros), Xssw(k, p, arma::fill::zeros);
  arma::vec zss(k, arma::fill::zeros);

  if (k < m) { // cw sketch
    arma::uvec selected_cols = arma::randi<arma::uvec>(m, arma::distr_param(0, k - 1));
    arma::ivec flip = arma::randi<arma::ivec>(m, arma::distr_param(0, 1)) * 2L - 1L;
    for (unsigned int j = 0; j < m; j++) {
      double fastmu = arma::as_scalar(X.row(j) * beta);
      fastmu = (tanh(fastmu/2) + 1)/2;
      zss(selected_cols(j)) += flip(j) * (y(j) - fastmu);
      fastmu = sqrt(fastmu * (1 - fastmu));
      fastmu = std::max(fastmu, arma::datum::eps);
      Xss.row(selected_cols(j)) += flip(j) * X.row(j);
      Xssw.row(selected_cols(j)) += flip(j) * X.row(j) * fastmu;
    }
  } else { // no cw sketch
    arma::vec mu = (arma::tanh(X * beta / 2) + 1)/2;
    arma::vec w = mu % (1 - mu);
    w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
    Xssw = X.each_col() % arma::sqrt(w);
    Xss = X;
    zss = (y - mu);
  }

  arma::mat hess_mat = Xssw.t() * Xssw;
  hess_mat.diag() += gamma / (sqrt(1 + iter));
  double a_m = arma::det(hess_mat);
  Fhat_m_num += a_m * arma::inv(hess_mat);
  Fhat_m_den += a_m;

  beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["k"] = k,
    Rcpp::_["gamma"] = gamma,
    Rcpp::_["iter"] = iter
  );
}

// [[Rcpp::export]]
Rcpp::List CW_Iter_Binomial_Logit_scale(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    arma::mat Fhat_m_num, double Fhat_m_den,
    const unsigned int k, const double gamma = 1, unsigned int iter = 1,
    const double scale_xz = 1
) {
  const unsigned int m = X.n_rows, p = X.n_cols;
  arma::mat Xss(k, p, arma::fill::zeros);
  arma::vec zss(k, arma::fill::zeros);
  
  if (k < m) { // cw sketch
    unsigned int csum = 0;
    for (unsigned int j = 0; j < k; j++) {
      unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
      if (rbinom == 0) continue;
      arma::mat Xs = X.rows(csum, csum + rbinom - 1);
      arma::vec ys = y.subvec(csum, csum + rbinom - 1);
      arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
      arma::vec w = (mu % (1 - mu));
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
      zss(j) = scale_xz * arma::accu(flip % ((ys - mu) / arma::sqrt(w)));
      Xss.row(j) = scale_xz * (flip % arma::sqrt(w)).t() * Xs;
      csum += rbinom;
      if (csum == m) break;
    }
  } else { // no cw sketch
    arma::vec mu = (arma::tanh(X * beta / 2) + 1)/2;
    arma::vec w = (mu % (1 - mu));
    w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
    Xss = scale_xz * (X.each_col() % arma::sqrt(w));
    zss = scale_xz * (y - mu) / arma::sqrt(w);
  }
  
  arma::mat hess_mat = Xss.t() * Xss;
  hess_mat.diag() += gamma / (sqrt(1 + iter));
  double a_m = arma::det(hess_mat);
  Fhat_m_num += a_m * arma::inv(hess_mat);
  Fhat_m_den += a_m;
  
  beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["k"] = k,
    Rcpp::_["gamma"] = gamma,
    Rcpp::_["iter"] = iter
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Poisson_Log(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;

  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;

  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {
      reset_iter += (++reset_length);
      Fhat_m_num.zeros();
      Fhat_m_den = 0;
    }
    arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
    arma::mat Xss(k, p, arma::fill::zeros), Xssw(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);

    if (k < m) { // cw sketch
      arma::uvec selected_cols = arma::randi<arma::uvec>(m, arma::distr_param(0, k - 1));
      arma::ivec flip = arma::randi<arma::ivec>(m, arma::distr_param(0, 1)) * 2L - 1L;
      for (unsigned int j = 0; j < m; j++) {
        double fastmu = arma::as_scalar(X.row(j) * beta);
        fastmu = exp(fastmu);
        zss(selected_cols(j)) += flip(j) * (y(j) - fastmu);
        Xss.row(selected_cols(j)) += flip(j) * X.row(j);
        Xssw.row(selected_cols(j)) += flip(j) * X.row(j) * fastmu;
      }
    } else { // no cw sketch
      arma::vec mu = arma::exp(X.rows(selected_rows) * beta);
      Xss = X.rows(selected_rows);
      Xssw = Xss.each_col() % arma::sqrt(mu);
      zss = (y.elem(selected_rows) - mu);
    }

    arma::mat hess_mat = Xssw.t() * Xssw;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;

    beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List CW_Iter_Poisson_Log(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    arma::mat Fhat_m_num, double Fhat_m_den,
    const unsigned int k, const double gamma = 1, unsigned int iter = 1
) {
  const unsigned int m = X.n_rows, p = X.n_cols;
  arma::mat Xss(k, p, arma::fill::zeros);
  arma::vec zss(k, arma::fill::zeros);
  
  if (k != m) { // cw sketch
    unsigned int csum = 0;
    for (unsigned int j = 0; j < k; j++) {
      unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
      if (rbinom == 0) continue;
      arma::mat Xs = X.rows(csum, csum + rbinom - 1);
      arma::vec ys = y.subvec(csum, csum + rbinom - 1);
      arma::vec w = arma::exp(Xs * beta);
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
      zss(j) = arma::accu(flip % ((ys - w) / arma::sqrt(w)));
      Xss.row(j) = (flip % arma::sqrt(w)).t() * Xs;
      csum += rbinom;
      if (csum == m) break;
    }
  } else { // no cw sketch
    arma::vec w = arma::exp(X * beta);
    w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
    Xss = X.each_col() % arma::sqrt(w);
    zss = (y - w) / arma::sqrt(w);
  }
  
  arma::mat hess_mat = Xss.t() * Xss;
  hess_mat.diag() += gamma / (sqrt(1 + iter));
  double a_m = arma::det(hess_mat);
  Fhat_m_num += a_m * arma::inv(hess_mat);
  Fhat_m_den += a_m;
  
  beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["k"] = k,
    Rcpp::_["gamma"] = gamma,
    Rcpp::_["iter"] = iter
  );
}

// [[Rcpp::export]]
Rcpp::List CW_Iter_Poisson_Log_shuffle(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    arma::mat Fhat_m_num, double Fhat_m_den,
    const unsigned int k, const double gamma = 1, unsigned int iter = 1
) {
  const unsigned int m = X.n_rows, p = X.n_cols;
  arma::mat Xss(k, p, arma::fill::zeros);
  arma::vec zss(k, arma::fill::zeros);
  
  if (k != m) { // cw sketch
    arma::uvec assign = arma::randi<arma::uvec>(m, arma::distr_param(0, k - 1));
    arma::uvec flip = arma::randi<arma::uvec>(m, arma::distr_param(0, 1));
    for (unsigned int j = 0; j < m; j++) {
      double sqrtw = arma::as_scalar(arma::clamp(arma::exp(0.5 * X.row(j) * beta), arma::datum::eps, arma::datum::inf));
      unsigned int idx = assign(j);
      if (flip(j) == 0L) {
        zss(idx) += y(j) / sqrtw - sqrtw;
        Xss.row(idx) += sqrtw * X.row(j);
      } else {
        zss(idx) -= y(j) / sqrtw - sqrtw;
        Xss.row(idx) -= sqrtw * X.row(j);
      }
    }
  } else { // no cw sketch
    arma::vec w = arma::exp(X * beta);
    w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
    Xss = X.each_col() % arma::sqrt(w);
    zss = (y - w) / arma::sqrt(w);
  }
  
  arma::mat hess_mat = Xss.t() * Xss;
  hess_mat.diag() += gamma / (sqrt(1 + iter));
  double a_m = arma::det(hess_mat);
  Fhat_m_num += a_m * arma::inv(hess_mat);
  Fhat_m_den += a_m;
  
  beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["k"] = k,
    Rcpp::_["gamma"] = gamma,
    Rcpp::_["iter"] = iter
  );
}

// [[Rcpp::export]]
Rcpp::List CW_Iter_Poisson_Log_scale(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    arma::mat Fhat_m_num, double Fhat_m_den,
    const unsigned int k, const double gamma = 1, unsigned int iter = 1,
    const double scale_xz = 1
) {
  const unsigned int m = X.n_rows, p = X.n_cols;
  arma::mat Xss(k, p, arma::fill::zeros);
  arma::vec zss(k, arma::fill::zeros);
  
  if (k < m) { // cw sketch
    unsigned int csum = 0;
    for (unsigned int j = 0; j < k; j++) {
      unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
      if (rbinom == 0) continue;
      arma::mat Xs = X.rows(csum, csum + rbinom - 1);
      arma::vec ys = y.subvec(csum, csum + rbinom - 1);
      arma::vec w = arma::exp(Xs * beta);
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
      zss(j) = scale_xz * arma::accu(flip % ((ys - w) / arma::sqrt(w)));
      Xss.row(j) = scale_xz * (flip % arma::sqrt(w)).t() * Xs;
      csum += rbinom;
      if (csum == m) break;
    }
  } else { // no cw sketch
    arma::vec w = arma::exp(X * beta);
    w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
    Xss = scale_xz * (X.each_col() % arma::sqrt(w));
    zss = scale_xz * (y - w) / arma::sqrt(w);
  }
  
  arma::mat hess_mat = Xss.t() * Xss;
  hess_mat.diag() += gamma / (sqrt(1 + iter));
  double a_m = arma::det(hess_mat);
  Fhat_m_num += a_m * arma::inv(hess_mat, arma::inv_opts::no_ugly);
  Fhat_m_den += a_m;
  
  beta = beta + (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["k"] = k,
    Rcpp::_["gamma"] = gamma,
    Rcpp::_["iter"] = iter
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Binomial_Cloglog(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros),Xssw(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        arma::mat Xs = Xx.rows(csum, csum + rbinom - 1);
        arma::vec ys = yy.subvec(csum, csum + rbinom - 1);
        arma::vec mu = -arma::expm1(-arma::exp(Xs * beta));
        mu = arma::clamp(mu, arma::datum::eps, 1 - arma::datum::eps);
        arma::vec w = (1 - mu) / mu % arma::square(arma::log1p(-mu));
        arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
        zss(j) = arma::accu(flip % ((ys - mu) / ((mu - 1) % arma::log1p(-mu)) % (w)));
        Xss.row(j) = (flip).t() * Xs;
        Xssw.row(j) = (flip % arma::sqrt(w)).t() * Xs;
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec mu = -arma::expm1(-arma::exp(Xs * beta));
      mu = arma::clamp(mu, arma::datum::eps, 1 - arma::datum::eps);
      arma::vec w = (1 - mu) / mu % arma::square(arma::log1p(-mu));
      Xss = Xs;
      Xssw = Xs.each_col() % arma::sqrt(w);
      zss = (ys - mu) / ((mu - 1) % arma::log1p(-mu)) % w;
    }
    
    arma::mat hess_mat = Xssw.t() * Xssw;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta += (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Binomial_Cloglog_full(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        arma::mat Xs = Xx.rows(csum, csum + rbinom - 1);
        arma::vec ys = yy.subvec(csum, csum + rbinom - 1);
        arma::vec mu = -arma::expm1(-arma::exp(Xs * beta));
        mu = arma::clamp(mu, arma::datum::eps, 1 - arma::datum::eps);
        arma::vec w = (1 - mu) / mu % arma::square(arma::log1p(-mu));
        arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
        zss(j) = arma::accu(flip % ((ys - mu) / ((mu - 1) % arma::log1p(-mu)) % arma::sqrt(w)));
        Xss.row(j) = (flip % arma::sqrt(w)).t() * Xs;
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec mu = -arma::expm1(-arma::exp(Xs * beta));
      mu = arma::clamp(mu, arma::datum::eps, 1 - arma::datum::eps);
      arma::vec w = (1 - mu) / mu % arma::square(arma::log1p(-mu));
      Xss = Xs.each_col() % arma::sqrt(w);
      zss = (ys - mu) / ((mu - 1) % arma::log1p(-mu)) % arma::sqrt(w);
    }
    
    arma::mat hess_mat = Xss.t() * Xss;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta += (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Binomial_Logit_full(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        arma::mat Xs = Xx.rows(csum, csum + rbinom - 1);
        arma::vec ys = yy.subvec(csum, csum + rbinom - 1);
        arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
        arma::vec w = (mu % (1 - mu));
        w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
        arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
        zss(j) = arma::accu(flip % ((ys - mu) / arma::sqrt(w)));
        Xss.row(j) = (flip % arma::sqrt(w)).t() * Xs;
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec mu = (arma::tanh(Xs * beta / 2) + 1)/2;
      arma::vec w = (mu % (1 - mu));
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      Xss = Xs.each_col() % arma::sqrt(w);
      zss = (ys - mu) / arma::sqrt(w);
    }
    
    arma::mat hess_mat = Xss.t() * Xss;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta += (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Poisson_Log_full(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat Xss(k, p, arma::fill::zeros);
    arma::vec zss(k, arma::fill::zeros);
    
    if (k < m) { // cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xx = X.rows(selected_rows);
      arma::vec yy = y.elem(selected_rows);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        arma::mat Xs = Xx.rows(csum, csum + rbinom - 1);
        arma::vec ys = yy.subvec(csum, csum + rbinom - 1);
        arma::vec w = arma::exp(Xs * beta);
        w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
        arma::ivec flip = arma::randi<arma::ivec>(rbinom, arma::distr_param(0, 1)) * 2L - 1L;
        zss(j) = arma::accu(flip % ((ys - w) / arma::sqrt(w)));
        Xss.row(j) = (flip % arma::sqrt(w)).t() * Xs;
        csum += rbinom;
        if (csum == m) break;
      }
    } else { // no cw sketch
      arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
      arma::mat Xs = X.rows(selected_rows);
      arma::vec ys = y.elem(selected_rows);
      arma::vec w = arma::exp(Xs * beta);
      w = arma::clamp(w, arma::datum::eps, arma::datum::inf);
      Xss = Xs.each_col() % arma::sqrt(w);
      zss = (ys - w) / arma::sqrt(w);
    }
    
    arma::mat hess_mat = Xss.t() * Xss;
    hess_mat.diag() += gamma / (sqrt(1 + iter));
    double a_m = arma::det(hess_mat);
    Fhat_m_num += a_m * arma::inv(hess_mat);
    Fhat_m_den += a_m;
    
    beta += (Fhat_m_num / Fhat_m_den) * Xss.t() * zss / (iter + 1);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den
  );
}

// [[Rcpp::export]]
Rcpp::List Double_Sketch_Gamma_Inverse_full(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10, const unsigned int backtrack = 100,
    const double gamma = 1,
    unsigned int reset_iter = 1, unsigned int reset_length = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat Fhat_m_num(p, p, arma::fill::zeros);
  double Fhat_m_den = 0;
  double dispersion = 0, lastdispersion = 0;
  
  for (unsigned int iter = 0; iter < iters; iter++) {
    if (iter == reset_iter) {reset_iter += (++reset_length); Fhat_m_num.zeros(); Fhat_m_den = 0;}
    arma::mat hess_mat;
    arma::vec diff;

    arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
    arma::mat Xx = X.rows(selected_rows);
    arma::vec yy = y.elem(selected_rows), eta = Xx * beta, zz = (1 - eta % yy);
    
    if (k < m) { // cw sketch
      arma::mat Xss(k, p, arma::fill::zeros);
      arma::vec zss(k, arma::fill::zeros);
      arma::ivec flip = arma::randi<arma::ivec>(m, arma::distr_param(0, 1)) * 2L - 1L;
      arma::vec cw_y = flip % zz;
      Xx.each_col() %= (flip / eta);
      
      unsigned int csum = 0;
      for (unsigned int j = 0; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0) continue;
        zss(j) = arma::accu(cw_y.subvec(csum, csum + rbinom - 1));
        Xss.row(j) = arma::sum(Xx.rows(csum, csum + rbinom - 1), 0);
        csum += rbinom;
        if (csum == m) break;
      }
      
      hess_mat = Xss.t() * Xss;
      hess_mat.diag() += gamma / (sqrt(1 + iter));
      double a_m = arma::det(hess_mat);
      Fhat_m_num += a_m * arma::inv(hess_mat);
      Fhat_m_den += a_m;
      diff = (Fhat_m_num / Fhat_m_den) * (Xss.t() * zss) / (iter + 1);
      lastdispersion = (n/m) * arma::accu(arma::square(zss)) / (n - p);
    } else { // no cw sketch
      Xx.each_col() /= eta;
      
      hess_mat = Xx.t() * Xx;
      hess_mat.diag() += gamma / (sqrt(1 + iter));
      double a_m = arma::det(hess_mat);
      Fhat_m_num += a_m * arma::inv(hess_mat);
      Fhat_m_den += a_m;
      diff = (Fhat_m_num / Fhat_m_den) * (Xx.t() * zz) / (iter + 1);
      lastdispersion = (n/m) * arma::accu(arma::square(zz)) / (n - p);
    }
    arma::vec Xdiff = Xx * diff;
    unsigned int inner_iter;
    for (inner_iter = 0; inner_iter < backtrack; inner_iter++) {
      arma::vec neweta = eta + Xdiff;
      if (neweta.is_finite() && arma::all(neweta > 0)) {break;}
      diff /= 2;
      Xdiff /= 2;
    }
    beta += diff;
    
    dispersion += (lastdispersion - dispersion) / (iter + 1.0);
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["Fhat_m_num"] = Fhat_m_num,
    Rcpp::_["Fhat_m_den"] = Fhat_m_den,
    Rcpp::_["dispersion"] = dispersion,
    Rcpp::_["lastdispersion"] = lastdispersion
  );
}
