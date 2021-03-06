!
! Copyright (C) Quantum ESPRESSO group
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
#if defined(__DFTI)
!=----------------------------------------------------------------------=!
   MODULE fft_scalar_dfti
!=----------------------------------------------------------------------=!

#if defined(__OPENMP_GPU)
       USE onemkl_dfti_omp_offload
       USE, intrinsic :: ISO_C_BINDING
#else
       USE onemkl_dfti
#endif
       USE fft_param

       IMPLICIT NONE
        SAVE

        PRIVATE
#if defined(__OPENMP_GPU)
        PUBLIC :: cft_1z_gpu, cft_2xy_gpu, cfft3d_gpu, cfft3ds_gpu
#endif
        PUBLIC :: cft_1z, cft_2xy, cfft3d, cfft3ds

        TYPE dfti_descriptor_array
           TYPE(DFTI_DESCRIPTOR), POINTER :: desc
        END TYPE

!=----------------------------------------------------------------------=!
   CONTAINS
!=----------------------------------------------------------------------=!

!
!=----------------------------------------------------------------------=!
!
!
!
!         FFT along "z"
!
!
!
!=----------------------------------------------------------------------=!
!

#if defined(__OPENMP_GPU)
   SUBROUTINE cft_1z_gpu(c, nsl, nz, ldz, isign, cout, in_place)

!     driver routine for nsl 1d complex fft's of length nz
!     ldz >= nz is the distance between sequences to be transformed
!     (ldz>nz is used on some architectures to reduce memory conflicts)
!     input  :  c(ldz*nsl)   (complex)
!     output : cout(ldz*nsl) (complex - NOTA BENE: transform is not in-place!)
!     isign > 0 : backward (f(G)=>f(R)), isign < 0 : forward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nz, nsl, ldz) are stored and re-used if available

     INTEGER, INTENT(IN)           :: isign
     INTEGER, INTENT(IN)           :: nsl, nz, ldz
     LOGICAL, INTENT(IN), OPTIONAL :: in_place

     COMPLEX (DP) :: c(:), cout(:)

     REAL (DP)  :: tscale
     INTEGER    :: i, err, idir, ip, void
     INTEGER, SAVE :: zdims( 3, ndims ) = -1
     INTEGER, SAVE :: icurrent = 1
     LOGICAL :: found

     INTEGER :: tid

#if defined(__OPENMP)
     INTEGER  :: offset, ldz_t
     INTEGER  :: omp_get_max_threads
     EXTERNAL :: omp_get_max_threads
#endif

     !   Intel MKL native FFT driver

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand( ndims )
     LOGICAL, SAVE :: dfti_first = .TRUE.
     LOGICAL, SAVE :: is_inplace
     INTEGER :: dfti_status = 0
     INTEGER :: placement

!$omp threadprivate(hand, dfti_first, dfti_status, zdims, icurrent, is_inplace)
     IF (PRESENT(in_place)) THEN
       is_inplace = in_place
     ELSE
       is_inplace = .false.
     endif
     !
     ! Check dimensions and corner cases.
     !
     IF ( nsl <= 0 ) THEN

       IF ( nsl < 0 ) CALL fftx_error__(" fft_scalar: cft_1z ", " nsl out of range ", nsl)

       ! Starting from MKL 2019 it is no longer possible to define "empty" plans,
       ! i.e. plans with 0 FFTs. Just return immediately in this case.
       RETURN

     END IF
     !
     !   Here initialize table only if necessary
     !
     CALL lookup()

     IF( .NOT. found ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !

#if defined(__FFT_CLOCKS)
     CALL start_clock( 'cft_1z' )
#endif

     IF (isign < 0) THEN
        IF (is_inplace) THEN
!$omp target variant dispatch use_device_ptr(c)
          dfti_status = DftiComputeForward(hand(ip)%desc, c )
!$omp end target variant dispatch
        ELSE
!$omp target variant dispatch use_device_ptr(c, cout)
          dfti_status = DftiComputeForward(hand(ip)%desc, c, cout )
!$omp end target variant dispatch
        ENDIF
        IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
     ELSE IF (isign > 0) THEN
        IF (is_inplace) THEN
!$omp target variant dispatch use_device_ptr(c)
          dfti_status = DftiComputeBackward(hand(ip)%desc, c)
!$omp end target variant dispatch
        ELSE
!$omp target variant dispatch use_device_ptr(c, cout)
          dfti_status = DftiComputeBackward(hand(ip)%desc, c, cout )
!$omp end target variant dispatch
        ENDIF
        IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU ',' stopped in DftiComputeBackward '// DftiErrorMessage(dfti_status), dfti_status )
     END IF

#if defined(__FFT_CLOCKS)
     CALL stop_clock( 'cft_1z' )
#endif

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE lookup()

     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     DO ip = 1, ndims
        !   first check if there is already a table initialized
        !   for this combination of parameters
        !   The initialization in ESSL and FFTW v.3 depends on all three parameters
        found = ( nz == zdims(1,ip) ) .AND. ( nsl == zdims(2,ip) ) .AND. ( ldz == zdims(3,ip) )
        dfti_status = DftiGetValue(hand(ip)%desc, DFTI_PLACEMENT, placement)
        found = found .AND. ( is_inplace .EQV. (placement == DFTI_INPLACE) )
        IF (found) EXIT
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()

       if( ASSOCIATED( hand( icurrent )%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand( icurrent )%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand( icurrent )%desc, DFTI_DOUBLE, DFTI_COMPLEX, 1,nz)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_NUMBER_OF_TRANSFORMS,nsl)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_INPUT_DISTANCE, ldz )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       IF (is_inplace) THEN
         dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       ELSE
         dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
       ENDIF
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_OUTPUT_DISTANCE, ldz )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_OUTPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/nz
       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_BACKWARD_SCALE, DBLE(1) );
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z GPU',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       !dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_THREAD_LIMIT, 1 );
       !IF(dfti_status /= 0) &
       !  CALL fftx_error__(' cft_1z ',' stopped in DFTI_THREAD_LIMIT ', dfti_status )

!$omp target variant dispatch
       dfti_status = DftiCommitDescriptor(hand( icurrent )%desc)
!$omp end target variant dispatch
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       zdims(1,icurrent) = nz; zdims(2,icurrent) = nsl; zdims(3,icurrent) = ldz;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1

     END SUBROUTINE init_dfti

   END SUBROUTINE cft_1z_gpu
#endif

   SUBROUTINE cft_1z(c, nsl, nz, ldz, isign, cout, in_place)

!     driver routine for nsl 1d complex fft's of length nz
!     ldz >= nz is the distance between sequences to be transformed
!     (ldz>nz is used on some architectures to reduce memory conflicts)
!     input  :  c(ldz*nsl)   (complex)
!     output : cout(ldz*nsl) (complex - NOTA BENE: transform is not in-place!)
!     isign > 0 : backward (f(G)=>f(R)), isign < 0 : forward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nz, nsl, ldz) are stored and re-used if available

     INTEGER, INTENT(IN)           :: isign
     INTEGER, INTENT(IN)           :: nsl, nz, ldz
     LOGICAL, INTENT(IN), OPTIONAL :: in_place

     COMPLEX (DP) :: c(:), cout(:)

     REAL (DP)  :: tscale
     INTEGER    :: i, err, idir, ip, void
     INTEGER, SAVE :: zdims( 3, ndims ) = -1
     INTEGER, SAVE :: icurrent = 1
     LOGICAL :: found

     INTEGER :: tid

#if defined(__OPENMP)
     INTEGER  :: offset, ldz_t
     INTEGER  :: omp_get_max_threads
     EXTERNAL :: omp_get_max_threads
#endif

     !   Intel MKL native FFT driver

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand( ndims )
     LOGICAL, SAVE :: dfti_first = .TRUE.
     LOGICAL, SAVE :: is_inplace
     INTEGER :: dfti_status = 0
     INTEGER :: placement

!$omp threadprivate(hand, dfti_first, dfti_status, zdims, icurrent, is_inplace)
     IF (PRESENT(in_place)) THEN
       is_inplace = in_place
     ELSE
       is_inplace = .false.
     endif
     !
     ! Check dimensions and corner cases.
     !
     IF ( nsl <= 0 ) THEN

       IF ( nsl < 0 ) CALL fftx_error__(" fft_scalar: cft_1z ", " nsl out of range ", nsl)

       ! Starting from MKL 2019 it is no longer possible to define "empty" plans,
       ! i.e. plans with 0 FFTs. Just return immediately in this case.
       RETURN

     END IF
     !
     !   Here initialize table only if necessary
     !
     CALL lookup()

     IF( .NOT. found ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !

#if defined(__FFT_CLOCKS)
     CALL start_clock( 'cft_1z' )
#endif

     IF (isign < 0) THEN
        IF (is_inplace) THEN
          dfti_status = DftiComputeForward(hand(ip)%desc, c )
        ELSE
          dfti_status = DftiComputeForward(hand(ip)%desc, c, cout )
        ENDIF
        IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
     ELSE IF (isign > 0) THEN
        IF (is_inplace) THEN
          dfti_status = DftiComputeBackward(hand(ip)%desc, c)
        ELSE
          dfti_status = DftiComputeBackward(hand(ip)%desc, c, cout )
        ENDIF
        IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DftiComputeBackward '// DftiErrorMessage(dfti_status), dfti_status )
     END IF

#if defined(__FFT_CLOCKS)
     CALL stop_clock( 'cft_1z' )
#endif

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE lookup()
     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     DO ip = 1, ndims
        !   first check if there is already a table initialized
        !   for this combination of parameters
        !   The initialization in ESSL and FFTW v.3 depends on all three parameters
        found = ( nz == zdims(1,ip) ) .AND. ( nsl == zdims(2,ip) ) .AND. ( ldz == zdims(3,ip) )
        dfti_status = DftiGetValue(hand(ip)%desc, DFTI_PLACEMENT, placement)
        found = found .AND. ( is_inplace .EQV. (placement == DFTI_INPLACE) )
        IF (found) EXIT
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()

       if( ASSOCIATED( hand( icurrent )%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand( icurrent )%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand( icurrent )%desc, DFTI_DOUBLE, DFTI_COMPLEX, 1,nz)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_NUMBER_OF_TRANSFORMS,nsl)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_INPUT_DISTANCE, ldz )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       IF (is_inplace) THEN
         dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       ELSE
         dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
       ENDIF
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_OUTPUT_DISTANCE, ldz )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_OUTPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/nz
       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_BACKWARD_SCALE, DBLE(1) );
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       !dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_THREAD_LIMIT, 1 );
       !IF(dfti_status /= 0) &
       !  CALL fftx_error__(' cft_1z ',' stopped in DFTI_THREAD_LIMIT ', dfti_status )

       dfti_status = DftiCommitDescriptor(hand( icurrent )%desc)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_1z ',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       zdims(1,icurrent) = nz; zdims(2,icurrent) = nsl; zdims(3,icurrent) = ldz;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1

     END SUBROUTINE init_dfti

   END SUBROUTINE cft_1z

!
!
!=----------------------------------------------------------------------=!
!
!
!
!         FFT along "x" and "y" direction
!
!
!
!=----------------------------------------------------------------------=!
!

#if defined(__OPENMP_GPU)
   SUBROUTINE cft_2xy_gpu(r, nzl, nx, ny, ldx, ldy, isign, pl2ix)

!     driver routine for nzl 2d complex fft's of lengths nx and ny
!     input : r(ldx*ldy)  complex, transform is in-place
!     ldx >= nx, ldy >= ny are the physical dimensions of the equivalent
!     2d array: r2d(ldx, ldy) (x first dimension, y second dimension)
!     (ldx>nx, ldy>ny used on some architectures to reduce memory conflicts)
!     pl2ix(nx) (optional) is 1 for columns along y to be transformed
!     isign > 0 : forward (f(G)=>f(R)), isign <0 backward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nx,ny,nzl,ldx) are stored and re-used if available

     IMPLICIT NONE

     INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl
     INTEGER, OPTIONAL, INTENT(IN) :: pl2ix(:)
     COMPLEX (DP), INTENT(INOUT) :: r( : )
     INTEGER :: i, k, j, err, idir, ip, kk, void
     REAL(DP) :: tscale
     INTEGER, SAVE :: icurrent = 1
     INTEGER, SAVE :: dims( 4, ndims) = -1
     LOGICAL :: dofft( nfftx ), found
     INTEGER, PARAMETER  :: stdout = 6

#if defined(__OPENMP)
     INTEGER :: offset
     INTEGER :: nx_t, ny_t, nzl_t, ldx_t, ldy_t
     INTEGER  :: itid, mytid, ntids
     INTEGER  :: omp_get_thread_num, omp_get_num_threads,omp_get_max_threads
     EXTERNAL :: omp_get_thread_num, omp_get_num_threads, omp_get_max_threads
#endif

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand( ndims )
     LOGICAL, SAVE :: dfti_first = .TRUE.
     INTEGER :: dfti_status = 0

!$omp threadprivate(hand, dfti_first, dfti_status, dims, icurrent)
     dofft( 1 : nx ) = .TRUE.
     IF( PRESENT( pl2ix ) ) THEN
       IF( SIZE( pl2ix ) < nx ) &
         CALL fftx_error__( ' cft_2xy ', ' wrong dimension for arg no. 8 ', 1 )
       DO i = 1, nx
         IF( pl2ix(i) < 1 ) dofft( i ) = .FALSE.
       END DO
     END IF

     !
     !   Here initialize table only if necessary
     !

     CALL lookup()

     IF( .NOT. found ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !

#if defined(__FFT_CLOCKS)
     CALL start_clock( 'cft_2xy' )
#endif

     IF( isign < 0 ) THEN
        !
!$omp target variant dispatch use_device_ptr(r)
        dfti_status = DftiComputeForward(hand(ip)%desc, r(:))
!$omp end target variant dispatch
        IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     ELSE IF( isign > 0 ) THEN
        !
!$omp target variant dispatch use_device_ptr(r)
        dfti_status = DftiComputeBackward(hand(ip)%desc, r(:))
!$omp end target variant dispatch
        IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU ',' stopped in DftiComputeBackward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     END IF

#if defined(__FFT_CLOCKS)
     CALL stop_clock( 'cft_2xy' )
#endif

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE check_dims()
     IF ( nx < 1 ) &
         call fftx_error__('cfft2d',' nx is less than 1 ', 1)
     IF ( ny < 1 ) &
         call fftx_error__('cfft2d',' ny is less than 1 ', 1)
     END SUBROUTINE check_dims

     SUBROUTINE lookup()
     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     DO ip = 1, ndims
       !   first check if there is already a table initialized
       !   for this combination of parameters
       found = ( ny == dims(1,ip) ) .AND. ( nx == dims(3,ip) )
       found = found .AND. ( ldx == dims(2,ip) ) .AND.  ( nzl == dims(4,ip) )
       IF (found) EXIT
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()

       if( ASSOCIATED( hand( icurrent )%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand( icurrent )%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand( icurrent )%desc, DFTI_DOUBLE, DFTI_COMPLEX, 2, [nx,ny])
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_NUMBER_OF_TRANSFORMS,nzl)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_INPUT_DISTANCE, ldx*ldy )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/ (nx * ny )
       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_BACKWARD_SCALE, DBLE(1) );
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

!$omp target variant dispatch
       dfti_status = DftiCommitDescriptor(hand( icurrent )%desc)
!$omp end target variant dispatch
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy GPU',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dims(1,icurrent) = ny; dims(2,icurrent) = ldx;
       dims(3,icurrent) = nx; dims(4,icurrent) = nzl;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1
     END SUBROUTINE init_dfti

   ENDSUBROUTINE cft_2xy_gpu
#endif

   SUBROUTINE cft_2xy(r, nzl, nx, ny, ldx, ldy, isign, pl2ix)

!     driver routine for nzl 2d complex fft's of lengths nx and ny
!     input : r(ldx*ldy)  complex, transform is in-place
!     ldx >= nx, ldy >= ny are the physical dimensions of the equivalent
!     2d array: r2d(ldx, ldy) (x first dimension, y second dimension)
!     (ldx>nx, ldy>ny used on some architectures to reduce memory conflicts)
!     pl2ix(nx) (optional) is 1 for columns along y to be transformed
!     isign > 0 : backward (f(G)=>f(R)), isign < 0 : forward (f(R) => f(G))
!     Up to "ndims" initializations (for different combinations of input
!     parameters nx,ny,nzl,ldx) are stored and re-used if available

     IMPLICIT NONE

     INTEGER, INTENT(IN) :: isign, ldx, ldy, nx, ny, nzl
     INTEGER, OPTIONAL, INTENT(IN) :: pl2ix(:)
     COMPLEX (DP), INTENT(INOUT) :: r( : )
     INTEGER :: i, k, j, err, idir, ip, kk, void
     REAL(DP) :: tscale
     INTEGER, SAVE :: icurrent = 1
     INTEGER, SAVE :: dims( 4, ndims) = -1
     LOGICAL :: dofft( nfftx ), found
     INTEGER, PARAMETER  :: stdout = 6

#if defined(__OPENMP)
     INTEGER :: offset
     INTEGER :: nx_t, ny_t, nzl_t, ldx_t, ldy_t
     INTEGER  :: itid, mytid, ntids
     INTEGER  :: omp_get_thread_num, omp_get_num_threads,omp_get_max_threads
     EXTERNAL :: omp_get_thread_num, omp_get_num_threads, omp_get_max_threads
#endif

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand( ndims )
     LOGICAL, SAVE :: dfti_first = .TRUE.
     INTEGER :: dfti_status = 0

!$omp threadprivate(hand, dfti_first, dfti_status, dims, icurrent)
     dofft( 1 : nx ) = .TRUE.
     IF( PRESENT( pl2ix ) ) THEN
       IF( SIZE( pl2ix ) < nx ) &
         CALL fftx_error__( ' cft_2xy ', ' wrong dimension for arg no. 8 ', 1 )
       DO i = 1, nx
         IF( pl2ix(i) < 1 ) dofft( i ) = .FALSE.
       END DO
     END IF

     !
     !   Here initialize table only if necessary
     !

     CALL lookup()

     IF( .NOT. found ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the FFTs using machine specific drivers
     !

#if defined(__FFT_CLOCKS)
     CALL start_clock( 'cft_2xy' )
#endif

     IF( isign < 0 ) THEN
        !
        dfti_status = DftiComputeForward(hand(ip)%desc, r(:))
        IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     ELSE IF( isign > 0 ) THEN
        !
        dfti_status = DftiComputeBackward(hand(ip)%desc, r(:))
        IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     END IF

#if defined(__FFT_CLOCKS)
     CALL stop_clock( 'cft_2xy' )
#endif

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE check_dims()
     IF ( nx < 1 ) &
         call fftx_error__('cfft2d',' nx is less than 1 ', 1)
     IF ( ny < 1 ) &
         call fftx_error__('cfft2d',' ny is less than 1 ', 1)
     END SUBROUTINE check_dims

     SUBROUTINE lookup()
     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     DO ip = 1, ndims
       !   first check if there is already a table initialized
       !   for this combination of parameters
       found = ( ny == dims(1,ip) ) .AND. ( nx == dims(3,ip) )
       found = found .AND. ( ldx == dims(2,ip) ) .AND.  ( nzl == dims(4,ip) )
       IF (found) EXIT
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()

       if( ASSOCIATED( hand( icurrent )%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand( icurrent )%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand( icurrent )%desc, DFTI_DOUBLE, DFTI_COMPLEX, 2, [nx,ny])
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_NUMBER_OF_TRANSFORMS,nzl)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc,DFTI_INPUT_DISTANCE, ldx*ldy )
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand( icurrent )%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/ (nx * ny )
       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue( hand( icurrent )%desc, DFTI_BACKWARD_SCALE, DBLE(1) );
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiCommitDescriptor(hand( icurrent )%desc)
       IF(dfti_status /= 0) CALL fftx_error__(' cft_2xy ',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dims(1,icurrent) = ny; dims(2,icurrent) = ldx;
       dims(3,icurrent) = nx; dims(4,icurrent) = nzl;
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1
     END SUBROUTINE init_dfti

   END SUBROUTINE cft_2xy

!
!=----------------------------------------------------------------------=!
!
!
!
!         3D scalar FFTs
!
!
!
!=----------------------------------------------------------------------=!
!

#if defined(__OPENMP_GPU)
   SUBROUTINE cfft3d_gpu( f, nx, ny, nz, ldx, ldy, ldz, howmany, isign )

  !     driver routine for 3d complex fft of lengths nx, ny, nz
  !     input  :  f(ldx*ldy*ldz)  complex, transform is in-place
  !     ldx >= nx, ldy >= ny, ldz >= nz are the physical dimensions
  !     of the equivalent 3d array: f3d(ldx,ldy,ldz)
  !     (ldx>nx, ldy>ny, ldz>nz may be used on some architectures
  !      to reduce memory conflicts - not implemented for FFTW)
  !     isign > 0 : f(G) => f(R)   ; isign < 0 : f(R) => f(G)
  !
  !     howmany: perform this many ffts, separated by ldx*ldy*ldz in memory
  !     Up to "ndims" initializations (for different combinations of input
  !     parameters nx,ny,nz) are stored and re-used if available

     IMPLICIT NONE

     INTEGER, INTENT(IN) :: nx, ny, nz, ldx, ldy, ldz, howmany, isign
     COMPLEX (DP) :: f(:)
     INTEGER :: i, k, j, err, idir, ip
     REAL(DP) :: tscale
     INTEGER, SAVE :: icurrent = 1
     INTEGER, SAVE :: dims(4,ndims) = -1

     !   Intel MKL native FFT driver

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand(ndims)
     LOGICAL, SAVE :: dfti_first = .TRUE.
     INTEGER :: dfti_status = 0
!$omp threadprivate(hand, dfti_first, dfti_status, dims, icurrent)
     !

     CALL check_dims()

     !
     !   Here initialize table only if necessary
     !

     CALL lookup()

     IF( ip == -1 ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the 3D FFT using the machine specific driver
     !

     IF( isign < 0 ) THEN
        !
        !$omp target variant dispatch use_device_ptr(f)
        dfti_status = DftiComputeForward(hand(ip)%desc, f(1:))
        !$omp end target variant dispatch
        IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     ELSE IF( isign > 0 ) THEN
        !
        !$omp target variant dispatch use_device_ptr(f)
        dfti_status = DftiComputeBackward(hand(ip)%desc, f(1:))
        !$omp end target variant dispatch
        IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU ',' stopped in DftiComputeBackward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     END IF

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE check_dims()
     IF ( nx < 1 ) &
         call fftx_error__('cfft3d',' nx is less than 1 ', 1)
     IF ( ny < 1 ) &
         call fftx_error__('cfft3d',' ny is less than 1 ', 1)
     IF ( nz < 1 ) &
         call fftx_error__('cfft3d',' nz is less than 1 ', 1)
     IF ( howmany < 1 ) &
         call fftx_error__('cfft3d',' howmany is less than 1 ', 1)
     END SUBROUTINE check_dims

     SUBROUTINE lookup()
     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     ip = -1
     DO i = 1, ndims
       !   first check if there is already a table initialized
       !   for this combination of parameters
       IF ( ( nx == dims(1,i) ) .and. &
            ( ny == dims(2,i) ) .and. &
            ( nz == dims(3,i) ) .and. &
            ( howmany == dims(4,i) ) ) THEN
         ip = i
         EXIT
       END IF
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()
      if( ASSOCIATED( hand(icurrent)%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand(icurrent)%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in cfft3d, DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand(icurrent)%desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, [nx,ny,nz])
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_NUMBER_OF_TRANSFORMS, howmany)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_INPUT_DISTANCE, ldx*ldy*ldz)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/ (nx * ny * nz)
       dfti_status = DftiSetValue( hand(icurrent)%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP
       dfti_status = DftiSetValue( hand(icurrent)%desc, DFTI_BACKWARD_SCALE, tscale );
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       !$omp target variant dispatch
       dfti_status = DftiCommitDescriptor(hand(icurrent)%desc)
       !$omp end target variant dispatch
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dims(1,icurrent) = nx; dims(2,icurrent) = ny; dims(3,icurrent) = nz; dims(4,icurrent) = howmany
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1
     END SUBROUTINE init_dfti

   END SUBROUTINE cfft3d_gpu
#endif

   SUBROUTINE cfft3d( f, nx, ny, nz, ldx, ldy, ldz, howmany, isign )

  !     driver routine for 3d complex fft of lengths nx, ny, nz
  !     input  :  f(ldx*ldy*ldz)  complex, transform is in-place
  !     ldx >= nx, ldy >= ny, ldz >= nz are the physical dimensions
  !     of the equivalent 3d array: f3d(ldx,ldy,ldz)
  !     (ldx>nx, ldy>ny, ldz>nz may be used on some architectures
  !      to reduce memory conflicts - not implemented for FFTW)
  !     isign > 0 : f(G) => f(R)   ; isign < 0 : f(R) => f(G)
  !
  !     howmany: perform this many ffts, separated by ldx*ldy*ldz in memory
  !     Up to "ndims" initializations (for different combinations of input
  !     parameters nx,ny,nz) are stored and re-used if available

     IMPLICIT NONE

     INTEGER, INTENT(IN) :: nx, ny, nz, ldx, ldy, ldz, howmany, isign
     COMPLEX (DP) :: f(:)
     INTEGER :: i, k, j, err, idir, ip
     REAL(DP) :: tscale
     INTEGER, SAVE :: icurrent = 1
     INTEGER, SAVE :: dims(4,ndims) = -1

     !   Intel MKL native FFT driver

     TYPE(DFTI_DESCRIPTOR_ARRAY), SAVE :: hand(ndims)
     LOGICAL, SAVE :: dfti_first = .TRUE.
     INTEGER :: dfti_status = 0
!$omp threadprivate(hand, dfti_first, dfti_status, dims, icurrent)
     !

     CALL check_dims()

     !
     !   Here initialize table only if necessary
     !

     CALL lookup()

     IF( ip == -1 ) THEN

       !   no table exist for these parameters
       !   initialize a new one

       CALL init_dfti()

     END IF

     !
     !   Now perform the 3D FFT using the machine specific driver
     !

     IF( isign < 0 ) THEN
        !
        dfti_status = DftiComputeForward(hand(ip)%desc, f(1:))
        IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DftiComputeForward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     ELSE IF( isign > 0 ) THEN
        !
        dfti_status = DftiComputeBackward(hand(ip)%desc, f(1:))
        IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DftiComputeBackward '// DftiErrorMessage(dfti_status), dfti_status )
        !
     END IF

     RETURN

   CONTAINS !=------------------------------------------------=!

     SUBROUTINE check_dims()
     IF ( nx < 1 ) &
         call fftx_error__('cfft3d',' nx is less than 1 ', 1)
     IF ( ny < 1 ) &
         call fftx_error__('cfft3d',' ny is less than 1 ', 1)
     IF ( nz < 1 ) &
         call fftx_error__('cfft3d',' nz is less than 1 ', 1)
     IF ( howmany < 1 ) &
         call fftx_error__('cfft3d',' howmany is less than 1 ', 1)
     END SUBROUTINE check_dims

     SUBROUTINE lookup()
     IF( dfti_first ) THEN
        DO ip = 1, ndims
           hand(ip)%desc => NULL()
        END DO
        dfti_first = .FALSE.
     END IF
     ip = -1
     DO i = 1, ndims
       !   first check if there is already a table initialized
       !   for this combination of parameters
       IF ( ( nx == dims(1,i) ) .and. &
            ( ny == dims(2,i) ) .and. &
            ( nz == dims(3,i) ) .and. &
            ( howmany == dims(4,i) ) ) THEN
         ip = i
         EXIT
       END IF
     END DO
     END SUBROUTINE lookup

     SUBROUTINE init_dfti()
      if( ASSOCIATED( hand(icurrent)%desc ) ) THEN
          dfti_status = DftiFreeDescriptor( hand(icurrent)%desc )
          IF( dfti_status /= 0) THEN
             WRITE(*,*) "stopped in cfft3d, DftiFreeDescriptor", dfti_status
             STOP
          ENDIF
       END IF

       dfti_status = DftiCreateDescriptor(hand(icurrent)%desc, DFTI_DOUBLE, DFTI_COMPLEX, 3, [nx,ny,nz])
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DftiCreateDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_NUMBER_OF_TRANSFORMS,howmany)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DFTI_NUMBER_OF_TRANSFORMS '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_INPUT_DISTANCE, ldx*ldy*ldz)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DFTI_INPUT_DISTANCE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiSetValue(hand(icurrent)%desc, DFTI_PLACEMENT, DFTI_INPLACE)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DFTI_PLACEMENT '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP/ (nx * ny * nz)
       dfti_status = DftiSetValue( hand(icurrent)%desc, DFTI_FORWARD_SCALE, tscale);
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DFTI_FORWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       tscale = 1.0_DP
       dfti_status = DftiSetValue( hand(icurrent)%desc, DFTI_BACKWARD_SCALE, tscale );
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d ',' stopped in DFTI_BACKWARD_SCALE '// DftiErrorMessage(dfti_status), dfti_status )

       dfti_status = DftiCommitDescriptor(hand(icurrent)%desc)
       IF(dfti_status /= 0) CALL fftx_error__(' cfft3d GPU',' stopped in DftiCommitDescriptor '// DftiErrorMessage(dfti_status), dfti_status )

       dims(1,icurrent) = nx; dims(2,icurrent) = ny; dims(3,icurrent) = nz; dims(4,icurrent) = howmany
       ip = icurrent
       icurrent = MOD( icurrent, ndims ) + 1
     END SUBROUTINE init_dfti

   END SUBROUTINE cfft3d

!
!=----------------------------------------------------------------------=!
!
!
!
!         3D scalar FFTs,  but using sticks!
!
!
!
!=----------------------------------------------------------------------=!
!

#if defined(__OPENMP_GPU)
   SUBROUTINE cfft3ds_gpu (f, nx, ny, nz, ldx, ldy, ldz, howmany, isign, do_fft_z, do_fft_y)
     !
     implicit none

     integer :: nx, ny, nz, ldx, ldy, ldz, isign, howmany
     !
     complex(DP) :: f ( ldx * ldy * ldz )
     integer :: do_fft_y(:), do_fft_z(:)
     !
     CALL cfft3d_gpu (f, nx, ny, nz, ldx, ldy, ldz, howmany, isign)

   END SUBROUTINE cfft3ds_gpu
#endif

   SUBROUTINE cfft3ds (f, nx, ny, nz, ldx, ldy, ldz, howmany, isign, do_fft_z, do_fft_y)
     !
     implicit none

     integer :: nx, ny, nz, ldx, ldy, ldz, isign, howmany
     !
     complex(DP) :: f ( ldx * ldy * ldz )
     integer :: do_fft_y(:), do_fft_z(:)
     !
     CALL cfft3d (f, nx, ny, nz, ldx, ldy, ldz, howmany, isign)

   END SUBROUTINE cfft3ds

#else
   MODULE fft_scalar_dfti
#endif
!=----------------------------------------------------------------------=!
END MODULE fft_scalar_dfti
!=----------------------------------------------------------------------=!
