// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Logistic_IRLS
Rcpp::List Logistic_IRLS(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int iters, const double eps);
RcppExport SEXP _doublysketchingglms_Logistic_IRLS(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_IRLS(X, y, beta, iters, eps));
    return rcpp_result_gen;
END_RCPP
}
// Logistic_Iter
arma::vec Logistic_Iter(const arma::mat& Xs, const arma::vec& ys, const arma::vec& beta);
RcppExport SEXP _doublysketchingglms_Logistic_Iter(SEXP XsSEXP, SEXP ysSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type Xs(XsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type ys(ysSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_Iter(Xs, ys, beta));
    return rcpp_result_gen;
END_RCPP
}
// Cloglog_IRLS
Rcpp::List Cloglog_IRLS(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int iters, const double eps);
RcppExport SEXP _doublysketchingglms_Cloglog_IRLS(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Cloglog_IRLS(X, y, beta, iters, eps));
    return rcpp_result_gen;
END_RCPP
}
// Poisson_IRLS
Rcpp::List Poisson_IRLS(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int iters, const double eps);
RcppExport SEXP _doublysketchingglms_Poisson_IRLS(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Poisson_IRLS(X, y, beta, iters, eps));
    return rcpp_result_gen;
END_RCPP
}
// Gamma_Inverse_IRLS
Rcpp::List Gamma_Inverse_IRLS(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int iters, const unsigned int backtrack, const double eps);
RcppExport SEXP _doublysketchingglms_Gamma_Inverse_IRLS(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP backtrackSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type backtrack(backtrackSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Gamma_Inverse_IRLS(X, y, beta, iters, backtrack, eps));
    return rcpp_result_gen;
END_RCPP
}
// Gamma_Inverse_wIRLS
Rcpp::List Gamma_Inverse_wIRLS(const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta, const unsigned int iters, const unsigned int backtrack, const double eps);
RcppExport SEXP _doublysketchingglms_Gamma_Inverse_wIRLS(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP backtrackSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type backtrack(backtrackSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Gamma_Inverse_wIRLS(X, y, w, beta, iters, backtrack, eps));
    return rcpp_result_gen;
END_RCPP
}
// Logistic_wIRLS
Rcpp::List Logistic_wIRLS(const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta, const unsigned int iters, const double eps);
RcppExport SEXP _doublysketchingglms_Logistic_wIRLS(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Logistic_wIRLS(X, y, w, beta, iters, eps));
    return rcpp_result_gen;
END_RCPP
}
// Poisson_wIRLS
Rcpp::List Poisson_wIRLS(const arma::mat& X, const arma::vec& y, const arma::vec& w, arma::vec beta, const unsigned int iters, const double eps);
RcppExport SEXP _doublysketchingglms_Poisson_wIRLS(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP, SEXP betaSEXP, SEXP itersSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type w(wSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(Poisson_wIRLS(X, y, w, beta, iters, eps));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Binomial_Logit
Rcpp::List Double_Sketch_Binomial_Logit(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Binomial_Logit(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Binomial_Logit(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Binomial_Logit_fold
Rcpp::List Double_Sketch_Binomial_Logit_fold(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Binomial_Logit_fold(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Binomial_Logit_fold(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// CW_Iter_Binomial_Logit
Rcpp::List CW_Iter_Binomial_Logit(const arma::mat& X, const arma::vec& y, arma::vec beta, arma::mat Fhat_m_num, double Fhat_m_den, const unsigned int k, const double gamma, unsigned int iter);
RcppExport SEXP _doublysketchingglms_CW_Iter_Binomial_Logit(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP Fhat_m_numSEXP, SEXP Fhat_m_denSEXP, SEXP kSEXP, SEXP gammaSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Fhat_m_num(Fhat_m_numSEXP);
    Rcpp::traits::input_parameter< double >::type Fhat_m_den(Fhat_m_denSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(CW_Iter_Binomial_Logit(X, y, beta, Fhat_m_num, Fhat_m_den, k, gamma, iter));
    return rcpp_result_gen;
END_RCPP
}
// CW_Iter_Binomial_Logit_scale
Rcpp::List CW_Iter_Binomial_Logit_scale(const arma::mat& X, const arma::vec& y, arma::vec beta, arma::mat Fhat_m_num, double Fhat_m_den, const unsigned int k, const double gamma, unsigned int iter, const double scale_xz);
RcppExport SEXP _doublysketchingglms_CW_Iter_Binomial_Logit_scale(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP Fhat_m_numSEXP, SEXP Fhat_m_denSEXP, SEXP kSEXP, SEXP gammaSEXP, SEXP iterSEXP, SEXP scale_xzSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Fhat_m_num(Fhat_m_numSEXP);
    Rcpp::traits::input_parameter< double >::type Fhat_m_den(Fhat_m_denSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< const double >::type scale_xz(scale_xzSEXP);
    rcpp_result_gen = Rcpp::wrap(CW_Iter_Binomial_Logit_scale(X, y, beta, Fhat_m_num, Fhat_m_den, k, gamma, iter, scale_xz));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Poisson_Log
Rcpp::List Double_Sketch_Poisson_Log(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Poisson_Log(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Poisson_Log(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// CW_Iter_Poisson_Log
Rcpp::List CW_Iter_Poisson_Log(const arma::mat& X, const arma::vec& y, arma::vec beta, arma::mat Fhat_m_num, double Fhat_m_den, const unsigned int k, const double gamma, unsigned int iter);
RcppExport SEXP _doublysketchingglms_CW_Iter_Poisson_Log(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP Fhat_m_numSEXP, SEXP Fhat_m_denSEXP, SEXP kSEXP, SEXP gammaSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Fhat_m_num(Fhat_m_numSEXP);
    Rcpp::traits::input_parameter< double >::type Fhat_m_den(Fhat_m_denSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(CW_Iter_Poisson_Log(X, y, beta, Fhat_m_num, Fhat_m_den, k, gamma, iter));
    return rcpp_result_gen;
END_RCPP
}
// CW_Iter_Poisson_Log_shuffle
Rcpp::List CW_Iter_Poisson_Log_shuffle(const arma::mat& X, const arma::vec& y, arma::vec beta, arma::mat Fhat_m_num, double Fhat_m_den, const unsigned int k, const double gamma, unsigned int iter);
RcppExport SEXP _doublysketchingglms_CW_Iter_Poisson_Log_shuffle(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP Fhat_m_numSEXP, SEXP Fhat_m_denSEXP, SEXP kSEXP, SEXP gammaSEXP, SEXP iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Fhat_m_num(Fhat_m_numSEXP);
    Rcpp::traits::input_parameter< double >::type Fhat_m_den(Fhat_m_denSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    rcpp_result_gen = Rcpp::wrap(CW_Iter_Poisson_Log_shuffle(X, y, beta, Fhat_m_num, Fhat_m_den, k, gamma, iter));
    return rcpp_result_gen;
END_RCPP
}
// CW_Iter_Poisson_Log_scale
Rcpp::List CW_Iter_Poisson_Log_scale(const arma::mat& X, const arma::vec& y, arma::vec beta, arma::mat Fhat_m_num, double Fhat_m_den, const unsigned int k, const double gamma, unsigned int iter, const double scale_xz);
RcppExport SEXP _doublysketchingglms_CW_Iter_Poisson_Log_scale(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP Fhat_m_numSEXP, SEXP Fhat_m_denSEXP, SEXP kSEXP, SEXP gammaSEXP, SEXP iterSEXP, SEXP scale_xzSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Fhat_m_num(Fhat_m_numSEXP);
    Rcpp::traits::input_parameter< double >::type Fhat_m_den(Fhat_m_denSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< const double >::type scale_xz(scale_xzSEXP);
    rcpp_result_gen = Rcpp::wrap(CW_Iter_Poisson_Log_scale(X, y, beta, Fhat_m_num, Fhat_m_den, k, gamma, iter, scale_xz));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Binomial_Cloglog
Rcpp::List Double_Sketch_Binomial_Cloglog(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Binomial_Cloglog(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Binomial_Cloglog(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Binomial_Cloglog_full
Rcpp::List Double_Sketch_Binomial_Cloglog_full(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Binomial_Cloglog_full(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Binomial_Cloglog_full(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Binomial_Logit_full
Rcpp::List Double_Sketch_Binomial_Logit_full(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Binomial_Logit_full(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Binomial_Logit_full(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Poisson_Log_full
Rcpp::List Double_Sketch_Poisson_Log_full(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Poisson_Log_full(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Poisson_Log_full(X, y, beta, m, k, iters, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// Double_Sketch_Gamma_Inverse_full
Rcpp::List Double_Sketch_Gamma_Inverse_full(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, const unsigned int backtrack, const double gamma, unsigned int reset_iter, unsigned int reset_length);
RcppExport SEXP _doublysketchingglms_Double_Sketch_Gamma_Inverse_full(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP backtrackSEXP, SEXP gammaSEXP, SEXP reset_iterSEXP, SEXP reset_lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type backtrack(backtrackSEXP);
    Rcpp::traits::input_parameter< const double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_iter(reset_iterSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type reset_length(reset_lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(Double_Sketch_Gamma_Inverse_full(X, y, beta, m, k, iters, backtrack, gamma, reset_iter, reset_length));
    return rcpp_result_gen;
END_RCPP
}
// SGD_Binomial_Logit
Rcpp::List SGD_Binomial_Logit(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, double learningrate);
RcppExport SEXP _doublysketchingglms_SGD_Binomial_Logit(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP learningrateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< double >::type learningrate(learningrateSEXP);
    rcpp_result_gen = Rcpp::wrap(SGD_Binomial_Logit(X, y, beta, m, k, iters, learningrate));
    return rcpp_result_gen;
END_RCPP
}
// SGD_Gamma_Inverse
Rcpp::List SGD_Gamma_Inverse(const arma::mat& X, const arma::vec& y, arma::vec beta, const unsigned int m, const unsigned int k, const unsigned int iters, double learningrate);
RcppExport SEXP _doublysketchingglms_SGD_Gamma_Inverse(SEXP XSEXP, SEXP ySEXP, SEXP betaSEXP, SEXP mSEXP, SEXP kSEXP, SEXP itersSEXP, SEXP learningrateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type m(mSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type k(kSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< double >::type learningrate(learningrateSEXP);
    rcpp_result_gen = Rcpp::wrap(SGD_Gamma_Inverse(X, y, beta, m, k, iters, learningrate));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_doublysketchingglms_Logistic_IRLS", (DL_FUNC) &_doublysketchingglms_Logistic_IRLS, 5},
    {"_doublysketchingglms_Logistic_Iter", (DL_FUNC) &_doublysketchingglms_Logistic_Iter, 3},
    {"_doublysketchingglms_Cloglog_IRLS", (DL_FUNC) &_doublysketchingglms_Cloglog_IRLS, 5},
    {"_doublysketchingglms_Poisson_IRLS", (DL_FUNC) &_doublysketchingglms_Poisson_IRLS, 5},
    {"_doublysketchingglms_Gamma_Inverse_IRLS", (DL_FUNC) &_doublysketchingglms_Gamma_Inverse_IRLS, 6},
    {"_doublysketchingglms_Gamma_Inverse_wIRLS", (DL_FUNC) &_doublysketchingglms_Gamma_Inverse_wIRLS, 7},
    {"_doublysketchingglms_Logistic_wIRLS", (DL_FUNC) &_doublysketchingglms_Logistic_wIRLS, 6},
    {"_doublysketchingglms_Poisson_wIRLS", (DL_FUNC) &_doublysketchingglms_Poisson_wIRLS, 6},
    {"_doublysketchingglms_Double_Sketch_Binomial_Logit", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Binomial_Logit, 9},
    {"_doublysketchingglms_Double_Sketch_Binomial_Logit_fold", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Binomial_Logit_fold, 9},
    {"_doublysketchingglms_CW_Iter_Binomial_Logit", (DL_FUNC) &_doublysketchingglms_CW_Iter_Binomial_Logit, 8},
    {"_doublysketchingglms_CW_Iter_Binomial_Logit_scale", (DL_FUNC) &_doublysketchingglms_CW_Iter_Binomial_Logit_scale, 9},
    {"_doublysketchingglms_Double_Sketch_Poisson_Log", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Poisson_Log, 9},
    {"_doublysketchingglms_CW_Iter_Poisson_Log", (DL_FUNC) &_doublysketchingglms_CW_Iter_Poisson_Log, 8},
    {"_doublysketchingglms_CW_Iter_Poisson_Log_shuffle", (DL_FUNC) &_doublysketchingglms_CW_Iter_Poisson_Log_shuffle, 8},
    {"_doublysketchingglms_CW_Iter_Poisson_Log_scale", (DL_FUNC) &_doublysketchingglms_CW_Iter_Poisson_Log_scale, 9},
    {"_doublysketchingglms_Double_Sketch_Binomial_Cloglog", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Binomial_Cloglog, 9},
    {"_doublysketchingglms_Double_Sketch_Binomial_Cloglog_full", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Binomial_Cloglog_full, 9},
    {"_doublysketchingglms_Double_Sketch_Binomial_Logit_full", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Binomial_Logit_full, 9},
    {"_doublysketchingglms_Double_Sketch_Poisson_Log_full", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Poisson_Log_full, 9},
    {"_doublysketchingglms_Double_Sketch_Gamma_Inverse_full", (DL_FUNC) &_doublysketchingglms_Double_Sketch_Gamma_Inverse_full, 10},
    {"_doublysketchingglms_SGD_Binomial_Logit", (DL_FUNC) &_doublysketchingglms_SGD_Binomial_Logit, 7},
    {"_doublysketchingglms_SGD_Gamma_Inverse", (DL_FUNC) &_doublysketchingglms_SGD_Gamma_Inverse, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_doublysketchingglms(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
