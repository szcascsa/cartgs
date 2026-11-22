#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper.h>
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/_linalg_eigh_meta.h>
#include <ATen/ops/_linalg_eigh_native.h>
#include <ATen/ops/_linalg_eigvals.h>
#include <ATen/ops/_linalg_eigvals_native.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/_lu_with_info_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/cholesky_inverse_native.h>
#include <ATen/ops/cholesky_native.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/cholesky_solve_native.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/geqrf_native.h>
#include <ATen/ops/inverse_native.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cholesky_ex_meta.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_cholesky_native.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eig_native.h>
#include <ATen/ops/linalg_eigh_native.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvals_native.h>
#include <ATen/ops/linalg_eigvalsh_native.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_householder_product_native.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_inv_native.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_factor_ex_meta.h>
#include <ATen/ops/linalg_ldl_factor_ex_native.h>
#include <ATen/ops/linalg_ldl_factor_native.h>
#include <ATen/ops/linalg_ldl_solve_meta.h>
#include <ATen/ops/linalg_ldl_solve_native.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_factor_ex_meta.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_meta.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_ex_native.h>
#include <ATen/ops/linalg_solve_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svd_native.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_svdvals_native.h>
#include <ATen/ops/linalg_vander_native.h>
#include <ATen/ops/linalg_vecdot_native.h>
#include <ATen/ops/lu_solve_native.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_meta.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/ormqr_native.h>
#include <ATen/ops/qr_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/svd_native.h>
#include <ATen/ops/triangular_solve_meta.h>
#include <ATen/ops/triangular_solve_native.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/zeros.h>
#endif

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#if AT_BUILD_WITH_LAPACK()

// getrf
extern "C" void zgetrf_(int *m, int *n, std::complex<double> *a, int *lda, int *ipiv, int *info);
extern "C" void cgetrf_(int *m, int *n, std::complex<float> *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);

// potrs
extern "C" void zpotrs_(char *uplo, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);
extern "C" void cpotrs_(char *uplo, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info);
extern "C" void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);

// potrf
extern "C" void zpotrf_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotrf_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

// potri
extern "C" void zpotri_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotri_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotri_(char *uplo, int *n, float *a, int *lda, int *info);

// sytrf
extern "C" void dsytrf_(
    char* uplo,
    int* n,
    double* a,
    int* lda,
    int* ipiv,
    double* work,
    int* lwork,
    int* info);
extern "C" void ssytrf_(
    char* uplo,
    int* n,
    float* a,
    int* lda,
    int* ipiv,
    float* work,
    int* lwork,
    int* info);
extern "C" void zsytrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void csytrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// hetrf
extern "C" void zhetrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void chetrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// sytrs
extern "C" void dsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    double* a,
    int* lda,
    int* ipiv,
    double* b,
    int* ldb,
    int* info);
extern "C" void ssytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    float* a,
    int* lda,
    int* ipiv,
    float* b,
    int* ldb,
    int* info);
extern "C" void zsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void csytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// hetrs
extern "C" void zhetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void chetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// geqrf
extern "C" void zgeqrf_(int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cgeqrf_(int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// orgqr
extern "C" void zungqr_(int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cungqr_(int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// ormqr
extern "C" void zunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *c, int *ldc, std::complex<double> *work, int *lwork, int *info);
extern "C" void cunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *c, int *ldc, std::complex<float> *work, int *lwork, int *info);
extern "C" void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern "C" void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);

// syevd
extern "C" void zheevd_(char *jobz, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void cheevd_(char *jobz, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work, int *lwork, float *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *iwork, int *liwork, int *info);
extern "C" void ssyevd_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *iwork, int *liwork, int *info);

// geev
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
extern "C" void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
extern "C" void cgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<float> *a, int *lda,
             std::complex<float> *w,
             std::complex<float> *vl, int *ldvl,
             std::complex<float> *vr, int *ldvr,
             std::complex<float> *work, int *lwork,
             float *rwork,
             int *info);
extern "C" void zgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<double> *a, int *lda,
             std::complex<double> *w,
             std::complex<double> *vl, int *ldvl,
             std::complex<double> *vr, int *ldvr,
             std::complex<double> *work, int *lwork,
             double *rwork,
             int *info);

// gesdd
extern "C" void zgesdd_(char *jobz, int *m, int *n, std::complex<double> *a, int *lda,
                        double *s, std::complex<double> *u, int *ldu, std::complex<double> *vt, int *ldvt, std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgesdd_(char *jobz, int *m, int *n, std::complex<float> *a, int *lda,
                        float *s, std::complex<float> *u, int *ldu, std::complex<float> *vt, int *ldvt, std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
                        double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info);
extern "C" void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
                        float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);

// getrs
extern "C" void zgetrs_(char *trans, int *n, int *nrhs, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *b, int *ldb, int *info);
extern "C" void cgetrs_(char *trans, int *n, int *nrhs, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *b, int *ldb, int *info);
extern "C" void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// gels
extern "C" void zgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    std::complex<double> *work, int *lwork, int *info);
extern "C" void cgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    std::complex<float> *work, int *lwork, int *info);
extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *work, int *lwork, int *info);
extern "C" void sgels_(char *trans, int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *work, int *lwork, int *info);

// gelsd
extern "C" void zgelsd_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgelsd_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgelsd_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *iwork, int *info);
extern "C" void sgelsd_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *iwork, int *info);

// gelsy
extern "C" void zgelsy_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelsy_(int *m, int *n, int *nrhs,
    std::complex<float> * a, int *lda, std::complex<float> *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelsy_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelsy_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    float *work, int *lwork, int *info);

// gelss
extern "C" void zgelss_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelss_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelss_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelss_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *info);
#endif

#if AT_BUILD_WITH_BLAS()
// trsm
extern "C" void ztrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<double> *alpha, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb);
extern "C" void ctrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<float> *alpha, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb);
extern "C" void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, double *alpha, double *a, int *lda, double *b, int *ldb);
extern "C" void strsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, float *alpha, float *a, int *lda, float *b, int *ldb);
#endif

namespace at::meta {

TORCH_META_FUNC(linalg_ldl_factor_ex)
(const Tensor& self, bool hermitian, bool check_errors) {
  at::native::squareCheckInputs(self, "torch.linalg.ldl_factor_ex");
  at::native::checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex");

  auto shape = self.sizes();
  auto ndim = shape.size();

  // prefer column major strides
  auto ld_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  set_output_strided(0, shape, ld_strides, self.options(), {}); // LD

  set_output_contiguous(
      1, shape.slice(0, ndim - 1), self.options().dtype(ScalarType::Int)); // pivots

  set_output_contiguous(
      2, shape.slice(0, ndim - 2), self.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_ldl_solve)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian) {
  at::native::squareCheckInputs(LD, "torch.linalg.ldl_solve");
  at::native::checkFloatingOrComplex(LD, "torch.linalg.ldl_solve");
  at::native::linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve");
  TORCH_CHECK(
      B.dim() >= 2,
      "torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, but it has ",
      B.dim(),
      " dimensions instead");
  auto expected_pivots_shape = LD.sizes().slice(0, LD.dim() - 1);
  TORCH_CHECK(
      expected_pivots_shape.equals(pivots.sizes()),
      "torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(),
      " instead");
  // pivots is allowed to be any integer type
  // LAPACK we use is 32-bit interface while cuSOLVER uses 64-bit interface for integers
  TORCH_CHECK(
      at::isIntegralType(pivots.scalar_type(), /*includeBool=*/false),
      "torch.linalg.ldl_solve: Expected pivots to be integers. Got ",
      pivots.scalar_type());
  TORCH_CHECK(
      LD.scalar_type() == B.scalar_type(),
      "torch.linalg.ldl_solve: ",
      "LD dtype",
      LD.scalar_type(),
      " does not match b dtype ",
      B.scalar_type());

    auto [B_broadcast_size, _] = at::native::_linalg_broadcast_batch_dims(B, LD);

  // prefer column major strides
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/true);
  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(triangular_solve)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular) {
  TORCH_CHECK(self.dim() >= 2,
           "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ", A.dim(), " dimensions instead");

  at::native::linearSolveCheckInputs(self, A, "triangular_solve");

  if (A.layout() == Layout::Strided) {
    auto [self_broadcast_size, A_broadcast_size] = at::native::_linalg_broadcast_batch_dims(self, A);

    // make column major strides for BLAS
    const auto solution_strides = at::native::batched_matrix_contiguous_strides(self_broadcast_size, /*f-contig=*/true);
    set_output_raw_strided(0, self_broadcast_size, solution_strides, self.options(), {});

    // make column major strides for BLAS
    auto clone_A_strides = at::native::batched_matrix_contiguous_strides(A_broadcast_size, /*f_contig=*/true);
    set_output_raw_strided(1, A_broadcast_size, clone_A_strides, A.options(), {});
  } else if (A.layout() == Layout::SparseCsr || A.layout() == Layout::SparseBsr) {
    // no broadcasting for non-strided layout
    set_output_raw_strided(0, self.sizes(), {}, self.options(), {}); // make row major strides for Sparse BLAS
    set_output_raw_strided(1, {0}, {}, self.options(), {}); // return 0-sized tensor
  } else {
    TORCH_INTERNAL_ASSERT(false, "triangular_solve: Got an unexpected layout.");
  }
}

TORCH_META_FUNC(_linalg_solve_ex)(const Tensor& A,
                                  const Tensor& B,
                                  bool left,
                                  bool check_errors) {
  // dtype
  at::native::checkFloatingOrComplex(A, "linalg.solve");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "linalg.solve: Expected A and B to have the same dtype, but found A of type ",
              A.scalar_type(), " and B of type ", B.scalar_type(), " instead");

  // NumPy compat: Two types of 'B' tensors are supported:
  // - 1D tensor or batch of 1D tensors (vector case)
  // - 2D tensor or batch of 2D tensors (matrix case)
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B);
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // matrix shapes
  at::native::checkInputsSolver(A, B_, /*left=*/left, "linalg.solve");

  // Check that B can be broadcasted to the shape of A
  auto B_broad_shape = std::get<0>(at::native::_linalg_broadcast_batch_dims(B_, A));
  // We disallow the broadcasting of B as a vector when left=False as, in that case, A.shape = (*, 1, 1)
  TORCH_CHECK(left || !vector_case, "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)");
  auto result_shape = vector_case ? IntArrayRef(B_broad_shape.data(), B_broad_shape.size() - 1)
                                  : B_broad_shape;
  auto result_strides = at::native::batched_matrix_contiguous_strides(result_shape, /*column_major=*/left);

  set_output_strided(0, result_shape, result_strides, B.options(), {});

  auto shape = A.sizes();
  auto ndim = shape.size();

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(1, shape, LU_strides, A.options(), {});

  // pivots
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));

  // info
  set_output_contiguous(3, shape.slice(0, ndim - 2), A.options().dtype(kInt));
}

TORCH_META_FUNC(linalg_inv_ex)(const Tensor& A, bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.inv");
  at::native::checkFloatingOrComplex(A, "linalg.inv", /*allow_low_precision_dtypes*/false);

  auto shape = A.sizes();

  auto result_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(0, shape, result_strides, A.options(), {});
  set_output_contiguous(
      1, shape.slice(0, shape.size() - 2), A.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_lu_factor_ex)(const Tensor& A, bool pivot, bool check_errors) {
  TORCH_CHECK(A.dim() >= 2, "torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];

  // make column major strides for BLAS
  auto LU_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
  set_output_strided(0, sizes, LU_strides, A.options(), {});

  // Set sizes to the size of pivots
  sizes.pop_back();
  sizes.back() = std::min(m, n);
  set_output_contiguous(1, sizes, A.options().dtype(kInt), {});

  // Set sizes to the size of info
  sizes.pop_back();
  set_output_contiguous(2, sizes, A.options().dtype(kInt), {});
}

TORCH_META_FUNC(linalg_lu_solve)(const Tensor& LU,
                                 const Tensor& pivots,
                                 const Tensor& B,
                                 bool left,
                                 bool adjoint) {
  // dtype
  at::native::checkFloatingOrComplex(LU, "torch.linalg.lu_solve");
  TORCH_CHECK(LU.scalar_type() == B.scalar_type(),
              "linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type ",
              LU.scalar_type(), " and B of type ", B.scalar_type(), " instead");
  TORCH_CHECK(pivots.dtype() == at::kInt,
              "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32");

  // matrix shapes
  at::native::squareCheckInputs(LU, "torch.linalg.lu_solve");
  at::native::checkInputsSolver(LU, B, left, "linalg.lu_solve");
  //
  TORCH_CHECK(LU.size(-1) == pivots.size(-1),
              "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix");

  // batches
  TORCH_CHECK(
      LU.sizes().slice(0, LU.dim() - 1).equals(pivots.sizes()),
      "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(), " instead");

  // This one checks that B can be broadcasted to the shape of A
  auto B_broadcast_size = std::get<0>(at::native::_linalg_broadcast_batch_dims(B, LU));
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/left);

  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(linalg_cholesky_ex)(const Tensor& A,
                                    bool upper,
                                    bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.cholesky");
  at::native::checkFloatingOrComplex(A, "linalg.cholesky");

  auto A_shape = A.sizes();
  auto ndim = A_shape.size();

  // L
  auto L_strides = at::native::batched_matrix_contiguous_strides(A_shape, /*f-contig*=*/true);
  set_output_strided(0, A_shape, L_strides, A.options(), {});

  // info
  set_output_contiguous(1, A_shape.slice(0, ndim - 2), A.options().dtype(ScalarType::Int));
}

TORCH_META_FUNC(linalg_qr)(const Tensor& A,
                           c10::string_view mode) {
  at::native::checkIsMatrix(A, "linalg.qr");
  at::native::checkFloatingOrComplex(A, "linalg.qr");
  auto [compute_q, reduced_mode] = at::native::_parse_qr_mode(mode);

  auto A_shape = A.sizes().vec();
  const auto m = A_shape.cend()[-2];
  const auto n = A_shape.cend()[-1];
  const auto k = std::min(m, n);

  if (compute_q) {
    auto Q_shape = A_shape;
    Q_shape.end()[-1] = reduced_mode ? k : m;
    auto Q_strides = at::native::batched_matrix_contiguous_strides(Q_shape, /*f-contig*=*/true);
    set_output_strided(0, Q_shape, Q_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // For readability
  auto R_shape = std::move(A_shape);
  R_shape.end()[-2] = (reduced_mode || !compute_q) ? k : m;
  auto R_strides = at::native::batched_matrix_contiguous_strides(R_shape, /*f-contig*=*/true);
  set_output_strided(1, R_shape, R_strides, A.options(), {});
}


TORCH_META_FUNC(_linalg_svd)(const Tensor& A,
                             bool full_matrices,
                             bool compute_uv,
                             c10::optional<c10::string_view> driver) {
  at::native::checkIsMatrix(A, "linalg.svd");
  at::native::checkFloatingOrComplex(A, "linalg.svd");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // Prepare sizes for U
  if (compute_uv) {
    sizes.back() = full_matrices ? m : k;
    auto U_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
    set_output_strided(0, sizes, U_strides, A.options(), {});

    // Prepare sizes for Vh
    sizes.end()[-2] = full_matrices ? n : k;
    sizes.end()[-1] = n;

    // We need to distinguish the cuSOLVER case, as the cuSOLVER algorithms we use
    // expect F-contig matrices, but they compute V rather than Vh
    const bool use_cusolver = at::native::svd_uses_cusolver(A);
    auto Vh_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/!use_cusolver);
    set_output_strided(2, sizes, Vh_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
    set_output_raw_strided(2, {0}, {}, A.options(), {});
  }

  // Prepare sizes for S. S is always real, even when A is complex.
  sizes.pop_back();
  sizes.end()[-1] = k;
  set_output_contiguous(1, sizes, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(lu_unpack)(const Tensor& LU, const Tensor& pivots, bool unpack_data, bool unpack_pivots) {
  TORCH_CHECK(LU.dim() >= 2, "torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: ", LU.sizes(), " instead");
  if (unpack_pivots) {
    TORCH_CHECK(pivots.scalar_type() == at::kInt,
        "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
        "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor");
  }

  auto sizes = LU.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (unpack_pivots) {
    set_output_raw_strided(0, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, LU.options(), {});
  }

  if (unpack_data) {
    // L.shape[-2:] == (m, k)
    sizes.end()[-1] = k;
    set_output_raw_strided(1, sizes, {}, LU.options(), {});

    // U.shape[-2:] == (k, n)
    sizes.end()[-2] = k;
    sizes.end()[-1] = n;
    set_output_raw_strided(2, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, LU.options(), {});
    set_output_raw_strided(2, {0}, {}, LU.options(), {});
  }
}

TORCH_META_FUNC(_linalg_eigh)(const Tensor& A,
                              c10::string_view uplo,
                              bool compute_v) {
  at::native::squareCheckInputs(A, "linalg.eigh");
  at::native::checkUplo(uplo);

  auto shape = A.sizes().vec();
  if (compute_v) {
    // eigenvectors
    auto V_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
    set_output_strided(1, shape, V_strides, A.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, A.options(), {});
  }

  // eigenvalues
  shape.pop_back();
  set_output_contiguous(0, shape, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(linalg_lu)(const Tensor& A, bool pivot) {
  TORCH_CHECK(A.dim() >= 2, "linalg.lu: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (pivot) {
    set_output_raw_strided(0, sizes, {}, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // L.shape[-2:] == (m, k)
  sizes.end()[-1] = k;
  set_output_raw_strided(1, sizes, {}, A.options(), {});

  // U.shape[-2:] == (k, n)
  sizes.end()[-2] = k;
  sizes.end()[-1] = n;
  set_output_raw_strided(2, sizes, {}, A.options(), {});
}

} // namespace at::meta

namespace at::native {

#if AT_BUILD_WITH_LAPACK()
// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations

template<class scalar_t>
void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info);

template<class scalar_t, class value_t=scalar_t>
void lapackSymeig(char jobz, char uplo, int n, scalar_t *a, int lda, value_t *w, scalar_t *work, int lwork, value_t *rwork, int *info);

template<> void lapackLu<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, int *ipiv, int *info) {
  zgetrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, int *ipiv, int *info) {
  cgetrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackCholeskySolve<c10::complex<double>>(char uplo, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, int *info) {
  zpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<c10::complex<float>>(char uplo, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, int *info) {
  cpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<double>(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholeskySolve<float>(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholesky<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotrf_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholesky<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotrf_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  spotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotri_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotri_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<float>(char uplo, int n, float *a, int lda, int *info) {
  spotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackGeqrf<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGeqrf<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGeqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackGeqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<c10::complex<double>>(int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zungqr_(&m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, 