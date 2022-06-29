MODULE LAXlib
#ifdef __CUDA
  USE cudafor
#endif
  IMPLICIT NONE
  PRIVATE
  !
  PUBLIC :: diaghg, pdiaghg
#if defined(__OPENMP_GPU)
  PUBLIC :: laxlib_cdiaghg_omp, laxlib_rdiaghg_omp

  INTERFACE
     SUBROUTINE laxlib_cdiaghg_omp( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
        USE laxlib_parallel_include
        USE onemkl_lapack_omp_offload
        INCLUDE 'laxlib_kinds.fh'
        INTEGER,     INTENT(IN) :: n, m, ldh
        COMPLEX(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
        REAL(DP),    INTENT(OUT) :: e(n)
        COMPLEX(DP), INTENT(OUT) :: v(ldh,m)
        INTEGER,     INTENT(IN) :: me_bgrp, root_bgrp, intra_bgrp_comm
     END SUBROUTINE laxlib_cdiaghg_omp

     SUBROUTINE laxlib_rdiaghg_omp( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
        USE laxlib_parallel_include
        USE onemkl_lapack_omp_offload
        INCLUDE 'laxlib_kinds.fh'
        INTEGER,  INTENT(IN) :: n, m, ldh
        REAL(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
        REAL(DP), INTENT(OUT) :: e(n)
        REAL(DP), INTENT(OUT) :: v(ldh,m)
        INTEGER,  INTENT(IN)  :: me_bgrp, root_bgrp, intra_bgrp_comm
     END SUBROUTINE laxlib_rdiaghg_omp
  END INTERFACE
#endif

  INTERFACE diaghg
    MODULE PROCEDURE cdiaghg_cpu_, rdiaghg_cpu_
    !
    SUBROUTINE laxlib_cdiaghg( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
       USE laxlib_parallel_include
       IMPLICIT NONE
       include 'laxlib_kinds.fh'
       INTEGER,     INTENT(IN) :: n, m, ldh
       COMPLEX(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
       REAL(DP),    INTENT(OUT) :: e(n)
       COMPLEX(DP), INTENT(OUT) :: v(ldh,m)
       INTEGER,     INTENT(IN)  :: me_bgrp, root_bgrp, intra_bgrp_comm
#if defined(__OPENMP_GPU)
       !$omp declare variant (laxlib_cdiaghg_omp) match( construct={dispatch} )
#endif
    END SUBROUTINE laxlib_cdiaghg
    !
    SUBROUTINE laxlib_rdiaghg( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
       USE laxlib_parallel_include
       IMPLICIT NONE
       include 'laxlib_kinds.fh'
       INTEGER,  INTENT(IN) :: n, m, ldh
       REAL(DP), INTENT(INOUT) :: h(ldh,n), s(ldh,n)
       REAL(DP), INTENT(OUT) :: e(n)
       REAL(DP), INTENT(OUT) :: v(ldh,m)
       INTEGER,  INTENT(IN)  :: me_bgrp, root_bgrp, intra_bgrp_comm
#if defined(__OPENMP_GPU)
       !$omp declare variant (laxlib_rdiaghg_omp) match( construct={dispatch} )
#endif
    END SUBROUTINE laxlib_rdiaghg
#ifdef __CUDA
    MODULE PROCEDURE cdiaghg_gpu_, rdiaghg_gpu_
    !
    SUBROUTINE laxlib_cdiaghg_gpu( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
#if defined(_OPENMP)
       USE omp_lib
#endif
#if defined(__CUDA)
       USE cudafor
       USE cusolverdn
#endif
       USE laxlib_parallel_include
#define __USE_GLOBAL_BUFFER
#if defined(__USE_GLOBAL_BUFFER) && defined(__CUDA)
       USE device_fbuff_m,        ONLY : dev=>dev_buf, pin=>pin_buf
#define VARTYPE POINTER
#else
#define VARTYPE ALLOCATABLE
#endif
       IMPLICIT NONE
       include 'laxlib_kinds.fh'
       INTEGER,     INTENT(IN) :: n, m, ldh
       COMPLEX(DP), INTENT(INOUT) :: h_d(ldh,n), s_d(ldh,n)
       REAL(DP),    INTENT(OUT) :: e_d(n)
       COMPLEX(DP), INTENT(OUT) :: v_d(ldh,n)
       INTEGER,     INTENT(IN)  :: me_bgrp, root_bgrp, intra_bgrp_comm
    ENDSUBROUTINE laxlib_cdiaghg_gpu
    !
    SUBROUTINE laxlib_rdiaghg_gpu( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
#if defined(_OPENMP)
       USE omp_lib
#endif
#if defined(__CUDA)
       USE cudafor
       USE cusolverdn
#endif
       USE laxlib_parallel_include
#define __USE_GLOBAL_BUFFER
#if defined(__USE_GLOBAL_BUFFER) && defined(__CUDA)
       USE device_fbuff_m,        ONLY : dev=>dev_buf, pin=>pin_buf
#define VARTYPE POINTER
#else
#define VARTYPE ALLOCATABLE
#endif
       IMPLICIT NONE
       include 'laxlib_kinds.fh'
       INTEGER,  INTENT(IN) :: n, m, ldh
       REAL(DP), INTENT(INOUT) :: h_d(ldh,n), s_d(ldh,n)
       REAL(DP), INTENT(OUT) :: e_d(n)
       REAL(DP), INTENT(OUT) :: v_d(ldh,n)
       INTEGER,  INTENT(IN)  :: me_bgrp, root_bgrp, intra_bgrp_comm
    ENDSUBROUTINE laxlib_rdiaghg_gpu
#endif
  END INTERFACE
  !
  INTERFACE pdiaghg
#if !defined(__OPENMP_GPU)
     MODULE PROCEDURE pcdiaghg_, prdiaghg_
#ifdef __CUDA
     MODULE PROCEDURE pcdiaghg__gpu, prdiaghg__gpu
#endif
#else
     MODULE PROCEDURE pcdiaghg, prdiaghg
#endif
  END INTERFACE
  !
  CONTAINS
  !
  !----------------------------------------------------------------------------
  SUBROUTINE cdiaghg_cpu_( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm, offload )
    !----------------------------------------------------------------------------
    !
    !! Called by diaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! complex matrices version.
    !! On output both matrix are unchanged.
    !!
    !! LAPACK version - uses both ZHEGV and ZHEGVX
    !
#if defined (__CUDA)
    USE cudafor
#endif
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized
    INTEGER, INTENT(IN) :: m
    !! number of eigenstates to be calculated
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    COMPLEX(DP), INTENT(INOUT) :: h(ldh,n)
    !! matrix to be diagonalized
    COMPLEX(DP), INTENT(INOUT) :: s(ldh,n)
    !! overlap matrix
    REAL(DP), INTENT(OUT) :: e(n)
    !! eigenvalues
    COMPLEX(DP), INTENT(OUT) :: v(ldh,m)
    !! eigenvectors (column-wise)
    INTEGER,  INTENT(IN)  :: me_bgrp
    !! index of the processor within a band group
    INTEGER,  INTENT(IN)  :: root_bgrp
    !! index of the root processor within a band group
    INTEGER,  INTENT(IN)  :: intra_bgrp_comm
    !! intra band group communicator
    LOGICAL, INTENT(IN) ::  offload
    !
#if defined(__CUDA)
    COMPLEX(DP), ALLOCATABLE, DEVICE :: v_d(:,:), h_d(:,:), s_d(:,:)
    REAL(DP),    ALLOCATABLE, DEVICE :: e_d(:)
    INTEGER :: info
#endif
    !
    ! the following ifdef ensures no offload if not compiling from GPU
    !
    IF ( offload ) THEN
#if defined(__CUDA)
      !
      ALLOCATE(s_d, source=s); ALLOCATE(h_d, source=h)
      ALLOCATE(e_d(n), v_d(ldh,n))
      !
      CALL laxlib_cdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
      !
      e = e_d
      v(1:ldh,1:m) = v_d(1:ldh,1:m)
      !
      DEALLOCATE(h_d, s_d, e_d, v_d)
#endif
    ELSE
      CALL laxlib_cdiaghg(n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm)
    END IF
    !
    RETURN
    !
  END SUBROUTINE cdiaghg_cpu_
  !
#if defined(__CUDA)
  !----------------------------------------------------------------------------
  SUBROUTINE cdiaghg_gpu_( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm, onhost )
    !----------------------------------------------------------------------------
    !
    !! Called by diaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! complex matrices version.
    !! On output both matrix are unchanged.
    !!
    !! GPU version
    !!
    !
    USE cudafor
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized
    INTEGER, INTENT(IN) :: m
    !! number of eigenstates to be calculate
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: h_d(ldh,n)
    !! matrix to be diagonalized
    COMPLEX(DP), DEVICE, INTENT(INOUT) :: s_d(ldh,n)
    !! overlap matrix
    REAL(DP), DEVICE, INTENT(OUT) :: e_d(n)
    !! eigenvalues
    COMPLEX(DP), DEVICE, INTENT(OUT) :: v_d(ldh,n)
    !! eigenvectors (column-wise)
    INTEGER,  INTENT(IN)  :: me_bgrp
    !! index of the processor within a band group
    INTEGER,  INTENT(IN)  :: root_bgrp
    !! index of the root processor within a band group
    INTEGER,  INTENT(IN)  :: intra_bgrp_comm
    !! intra band group communicator
    LOGICAL, INTENT(IN) ::  onhost
    !
    COMPLEX(DP), ALLOCATABLE :: v(:,:), h(:,:), s(:,:)
    REAL(DP),    ALLOCATABLE :: e(:)
    !
    INTEGER :: info
    !
    !
    IF ( onhost ) THEN
      !
      ALLOCATE(s, source=s_d); ALLOCATE(h, source=h_d)
      ALLOCATE(e(n), v(ldh,m))
      !
      CALL laxlib_cdiaghg(n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm)
      !
      e_d = e
      v_d(1:ldh,1:m) = v(1:ldh,1:m)
      !
      DEALLOCATE(h, s, e, v)
    ELSE
      CALL laxlib_cdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
    END IF
    !
    RETURN
    !
  END SUBROUTINE cdiaghg_gpu_
#endif
  !
  !----------------------------------------------------------------------------
  SUBROUTINE rdiaghg_cpu_( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm, offload )
    !----------------------------------------------------------------------------
    !
    !! Called by diaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! real matrices version.
    !! On output both matrix are unchanged.
    !!
    !! LAPACK version - uses both DSYGV and DSYGVX
    !!
    !
#if defined(__CUDA)
    USE cudafor
#endif
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized
    INTEGER, INTENT(IN) :: m
    !! number of eigenstates to be calculate
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    REAL(DP), INTENT(INOUT) :: h(ldh,n)
    !! matrix to be diagonalized
    REAL(DP), INTENT(INOUT) :: s(ldh,n)
    !! overlap matrix
    REAL(DP), INTENT(OUT) :: e(n)
    !! eigenvalues
    REAL(DP), INTENT(OUT) :: v(ldh,m)
    !! eigenvectors (column-wise)
    INTEGER,  INTENT(IN)  :: me_bgrp
    !! index of the processor within a band group
    INTEGER,  INTENT(IN)  :: root_bgrp
    !! index of the root processor within a band group
    INTEGER,  INTENT(IN)  :: intra_bgrp_comm
    !! intra band group communicator
    LOGICAL, INTENT(IN) ::  offload
    !
#if defined(__CUDA)
    REAL(DP), ALLOCATABLE, DEVICE :: v_d(:,:), h_d(:,:), s_d(:,:)
    REAL(DP), ALLOCATABLE, DEVICE :: e_d(:)
    INTEGER :: info
#endif
    !
    ! the following ifdef ensures no offload if not compiling from GPU
    !
    IF ( offload ) THEN
#if defined(__CUDA)
      !
      ALLOCATE(s_d, source=s); ALLOCATE(h_d, source=h)
      ALLOCATE(e_d(n), v_d(ldh,n))
      !
      CALL laxlib_rdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
      !
      e = e_d
      v(1:ldh,1:m) = v_d(1:ldh,1:m)
      !
      DEALLOCATE(h_d, s_d, e_d, v_d)
#endif
    ELSE
      CALL laxlib_rdiaghg(n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm)
    END IF
    !
    RETURN
    !
  END SUBROUTINE rdiaghg_cpu_
  !
#if defined(__CUDA)
  !----------------------------------------------------------------------------
  SUBROUTINE rdiaghg_gpu_( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm, onhost )
    !----------------------------------------------------------------------------
    !
    !! Called by diaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! real matrices version.
    !! On output both matrix are unchanged.
    !!
    !! GPU version
    !!
    !
    USE cudafor
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized
    INTEGER, INTENT(IN) :: m
    !! number of eigenstates to be calculate
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    REAL(DP), DEVICE, INTENT(INOUT) :: h_d(ldh,n)
    !! matrix to be diagonalized
    REAL(DP), DEVICE, INTENT(INOUT) :: s_d(ldh,n)
    !! overlap matrix
    REAL(DP), DEVICE, INTENT(OUT) :: e_d(n)
    !! eigenvalues
    REAL(DP), DEVICE, INTENT(OUT) :: v_d(ldh,n)
    !! eigenvectors (column-wise)
    INTEGER,  INTENT(IN)  :: me_bgrp
    !! index of the processor within a band group
    INTEGER,  INTENT(IN)  :: root_bgrp
    !! index of the root processor within a band group
    INTEGER,  INTENT(IN)  :: intra_bgrp_comm
    !! intra band group communicator
    LOGICAL, INTENT(IN) ::  onhost
    !
    REAL(DP), ALLOCATABLE :: v(:,:), h(:,:), s(:,:)
    REAL(DP), ALLOCATABLE :: e(:)
    !
    INTEGER :: info
    !
    !
    IF ( onhost ) THEN
      !
      ALLOCATE(s, source=s_d); ALLOCATE(h, source=h_d)
      ALLOCATE(e(n), v(ldh,m))
      !
      CALL laxlib_rdiaghg(n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm)
      !
      e_d = e
      v_d(1:ldh,1:m) = v(1:ldh,1:m)
      !
      DEALLOCATE(h, s, e, v)
    ELSE
      CALL laxlib_rdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
    END IF
    !
    RETURN
    !
  END SUBROUTINE rdiaghg_gpu_
#endif
  !
  !  === Parallel diagonalization interface subroutines
  !
  !
#if !defined(__OPENMP_GPU)
  !----------------------------------------------------------------------------
  SUBROUTINE prdiaghg_( n, h, s, ldh, e, v, idesc, offload )
    !----------------------------------------------------------------------------
    !
    !! Called by pdiaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! real matrices version.
    !! On output both matrix are unchanged.
    !!
    !! Parallel version with full data distribution
    !!
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    REAL(DP), INTENT(INOUT) :: h(ldh,ldh)
    !! matrix to be diagonalized
    REAL(DP), INTENT(INOUT) :: s(ldh,ldh)
    !! overlap matrix
    REAL(DP), INTENT(OUT) :: e(n)
    !! eigenvalues
    REAL(DP), INTENT(OUT) :: v(ldh,ldh)
    !! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
    !! laxlib descriptor
    LOGICAL, OPTIONAL ::  offload
    !! place-holder, offloading on GPU not implemented yet
    LOGICAL :: loffload

    CALL laxlib_prdiaghg( n, h, s, ldh, e, v, idesc)

  END SUBROUTINE
  !----------------------------------------------------------------------------
  SUBROUTINE pcdiaghg_( n, h, s, ldh, e, v, idesc, offload )
    !----------------------------------------------------------------------------
    !
    !! Called by pdiaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! complex matrices version.
    !! On output both matrix are unchanged.
    !!
    !! Parallel version with full data distribution
    !!
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    COMPLEX(DP), INTENT(INOUT) :: h(ldh,ldh)
    !! matrix to be diagonalized
    COMPLEX(DP), INTENT(INOUT) :: s(ldh,ldh)
    !! overlap matrix
    REAL(DP), INTENT(OUT) :: e(n)
    !! eigenvalues
    COMPLEX(DP), INTENT(OUT) :: v(ldh,ldh)
    !! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
    !! laxlib descriptor
    LOGICAL, OPTIONAL ::  offload
    !! place-holder, offloading on GPU not implemented yet
    LOGICAL :: loffload

    CALL laxlib_pcdiaghg( n, h, s, ldh, e, v, idesc)

  END SUBROUTINE
  !
#if defined(__CUDA)
  !----------------------------------------------------------------------------
  SUBROUTINE prdiaghg__gpu( n, h_d, s_d, ldh, e_d, v_d, idesc, onhost )
    !----------------------------------------------------------------------------
    !
    !! Called by pdiaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! real matrices version.
    !! On output both matrix are unchanged.
    !!
    !! Parallel GPU version with full data distribution
    !!
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    REAL(DP), INTENT(INOUT), DEVICE :: h_d(ldh,ldh)
    !! matrix to be diagonalized
    REAL(DP), INTENT(INOUT), DEVICE :: s_d(ldh,ldh)
    !! overlap matrix
    REAL(DP), INTENT(OUT), DEVICE :: e_d(n)
    !! eigenvalues
    REAL(DP), INTENT(OUT), DEVICE :: v_d(ldh,ldh)
    !! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
    !! laxlib descriptor
    LOGICAL, OPTIONAL ::  onhost
    !! place-holder, prdiaghg on GPU not implemented yet
    LOGICAL :: lonhost
    !
    REAL(DP), ALLOCATABLE :: v(:,:), h(:,:), s(:,:)
    REAL(DP), ALLOCATABLE :: e(:)

    ALLOCATE(h(ldh,ldh), s(ldh,ldh), e(n), v(ldh,ldh))
    h = h_d; s = s_d;
    CALL laxlib_prdiaghg( n, h, s, ldh, e, v, idesc)
    e_d = e; v_d = v
    DEALLOCATE(h,s,v,e)
    !
  END SUBROUTINE
  !----------------------------------------------------------------------------
  SUBROUTINE pcdiaghg__gpu( n, h_d, s_d, ldh, e_d, v_d, idesc, onhost )
    !----------------------------------------------------------------------------
    !
    !! Called by pdiaghg interface.
    !! Calculates eigenvalues and eigenvectors of the generalized problem.
    !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
    !! complex matrices version.
    !! On output both matrix are unchanged.
    !!
    !! Parallel GPU version with full data distribution
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n
    !! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
    INTEGER, INTENT(IN) :: ldh
    !! leading dimension of h, as declared in the calling pgm unit
    COMPLEX(DP), INTENT(INOUT), DEVICE :: h_d(ldh,ldh)
    !! matrix to be diagonalized
    COMPLEX(DP), INTENT(INOUT), DEVICE :: s_d(ldh,ldh)
    !! overlap matrix
    REAL(DP), INTENT(OUT), DEVICE :: e_d(n)
    !! eigenvalues
    COMPLEX(DP), INTENT(OUT), DEVICE :: v_d(ldh,ldh)
    !! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
    !! laxlib descriptor
    LOGICAL, OPTIONAL ::  onhost
    !! place-holder, pcdiaghg on GPU not implemented yet
    LOGICAL :: lonhost
      !
    COMPLEX(DP), ALLOCATABLE :: v(:,:), h(:,:), s(:,:)
    REAL(DP), ALLOCATABLE :: e(:)

    ALLOCATE(h(ldh,ldh), s(ldh,ldh), e(n), v(ldh,ldh))
    h = h_d; s = s_d;
    CALL laxlib_pcdiaghg( n, h, s, ldh, e, v, idesc)
    e_d = e; v_d = v
    DEALLOCATE(h,s,v,e)
    !
  END SUBROUTINE
#endif
  !
#else
  !----------------------------------------------------------------------------
  SUBROUTINE prdiaghg( n, h, s, ldh, e, v, idesc, offload )
    !----------------------------------------------------------------------------
    !
    ! ... calculates eigenvalues and eigenvectors of the generalized problem
    ! ... Hv=eSv, with H symmetric matrix, S overlap matrix.
    ! ... On output both matrix are unchanged
    !
    ! ... Parallel version with full data distribution
    !
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n, ldh
      ! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
      ! leading dimension of h, as declared in the calling pgm unit
    REAL(DP), INTENT(INOUT) :: h(ldh,ldh), s(ldh,ldh)
      ! matrix to be diagonalized
      ! overlap matrix
    !
    REAL(DP), INTENT(OUT) :: e(n)
      ! eigenvalues
    REAL(DP), INTENT(OUT) :: v(ldh,ldh)
      ! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
      !
    LOGICAL, INTENT(IN) ::  offload
      ! place-holder, prdiaghg on GPU not implemented yet
      !
    IF (offload) THEN
       !$omp target update from(h, s)
       CALL laxlib_prdiaghg( n, h, s, ldh, e, v, idesc)
       !$omp target update to(e, v)
    ELSE
       CALL laxlib_prdiaghg( n, h, s, ldh, e, v, idesc)
    ENDIF
    !
  END SUBROUTINE
  !----------------------------------------------------------------------------
  SUBROUTINE pcdiaghg( n, h, s, ldh, e, v, idesc, offload )
    !----------------------------------------------------------------------------
    !
    ! ... calculates eigenvalues and eigenvectors of the generalized problem
    ! ... Hv=eSv, with H symmetric matrix, S overlap matrix.
    ! ... On output both matrix are unchanged
    !
    ! ... Parallel version with full data distribution
    !
    USE iso_c_binding, ONLY : c_ptr
    IMPLICIT NONE
    include 'laxlib_kinds.fh'
    include 'laxlib_param.fh'
    !
    INTEGER, INTENT(IN) :: n, ldh
      ! dimension of the matrix to be diagonalized and number of eigenstates to be calculated
      ! leading dimension of h, as declared in the calling pgm unit
    COMPLEX(DP), INTENT(INOUT) :: h(ldh,ldh), s(ldh,ldh)
      ! matrix to be diagonalized
      ! overlap matrix
    !
    REAL(DP), INTENT(OUT) :: e(n)
      ! eigenvalues
    COMPLEX(DP), INTENT(OUT) :: v(ldh,ldh)
      ! eigenvectors (column-wise)
    INTEGER, INTENT(IN) :: idesc(LAX_DESC_SIZE)
      !
    LOGICAL, INTENT(IN) ::  offload
      !
    IF (offload) THEN
       !$omp target update from(h, s)
       CALL laxlib_pcdiaghg( n, h, s, ldh, e, v, idesc)
       !$omp target update to(e, v)
    ELSE
       CALL laxlib_pcdiaghg( n, h, s, ldh, e, v, idesc)
    ENDIF
    !
  END SUBROUTINE
#endif
END MODULE LAXlib
