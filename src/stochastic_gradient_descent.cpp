// -*- mode: C++; c-indent-level: 2; c-basic-offset: 2; indent-tabs-mode: nil; -*-
// #define ARMA_WARN_LEVEL 1
// #define ARMA_NO_DEBUG
// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export]]
Rcpp::List SGD_Binomial_Logit(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10L, double learningrate = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::zeros);
  learningrate *= m;
  learningrate /= n;
  
  for (unsigned int iter = 0L; iter < iters; iter++) {
    arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
    arma::mat Xx = X.rows(selected_rows);
    arma::vec mu = (arma::tanh(Xx * beta / 2) + 1)/2;
    arma::vec yy = y.elem(selected_rows) - mu; // yy now stores (yy - mu) 
    
    if (k < m) { // cw sketch
      arma::mat Xss(k, p, arma::fill::none);
      arma::vec zss(k, arma::fill::none);
      arma::vec flip = arma::randi<arma::vec>(m, arma::distr_param(0, 1)) * 2L - 1L;
      Xx.each_col() %= flip;
      yy %= flip;
      
      unsigned int csum = 0L;
      for (unsigned int j = 0L; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0L) {
          zss(j) = 0;
          Xss.row(j).zeros();
        } else {
          zss(j) = arma::accu(yy.subvec(csum, csum + rbinom - 1L));
          Xss.row(j) = arma::sum(Xx.rows(csum, csum + rbinom - 1L), 0);
          csum += rbinom;
          if (csum == m) break;
        }
      }
      beta = beta + Xss.t() * zss * learningrate / sqrt(iter + 1.0);
    } else { // no cw sketch
      beta = beta + Xx.t() * yy * learningrate / sqrt(iter + 1.0);
    }
    B.row(iter) = beta.t();
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["B"] = B
  );
}

// [[Rcpp::export]]
Rcpp::List SGD_Gamma_Inverse(
    const arma::mat& X, const arma::vec& y, arma::vec beta,
    const unsigned int m, const unsigned int k,
    const unsigned int iters = 10L, double learningrate = 1
) {
  const unsigned int n = X.n_rows, p = X.n_cols;
  arma::mat B(iters, p, arma::fill::zeros);
  learningrate *= m;
  learningrate /= n;
  
  for (unsigned int iter = 0L; iter < iters; iter++) {
    arma::uvec selected_rows = arma::randi<arma::uvec>(m, arma::distr_param(0, n - 1));
    arma::mat Xx = X.rows(selected_rows);
    arma::vec mu = (arma::tanh(Xx * beta / 2) + 1)/2;
    arma::vec yy = y.elem(selected_rows) - mu, eta = Xx * beta, zz = (1 - eta % yy);
    
    if (k < m) { // cw sketch
      arma::mat Xss(k, p, arma::fill::zeros);
      arma::vec zss(k, arma::fill::zeros);
      arma::vec flip = arma::randi<arma::vec>(m, arma::distr_param(0, 1)) * 2L - 1L;
      arma::vec cw_y = flip % zz;
      Xx.each_col() %= (flip / eta);
      
      unsigned int csum = 0L;
      for (unsigned int j = 0L; j < k; j++) {
        unsigned int rbinom = R::rbinom(m - csum, 1.0/(k - j));
        if (rbinom == 0L) continue;
        zss(j) = arma::accu(cw_y.subvec(csum, csum + rbinom - 1L));
        Xss.row(j) = arma::sum(Xx.rows(csum, csum + rbinom - 1L), 0);
        csum += rbinom;
        if (csum == m) break;
      }
      beta = beta + Xss.t() * zss * learningrate / (iter + 1.0);
    } else { // no cw sketch
      beta = beta + Xx.t() * yy * learningrate / (iter + 1.0);
    }
    B.row(iter) = beta.t();
    if (beta.has_nan()) {break;}
  }
  return Rcpp::List::create(
    Rcpp::_["beta"] = beta,
    Rcpp::_["B"] = B
  );
}
