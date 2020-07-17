!
! Copyright (C) 2002-2011 Quantum ESPRESSO groups
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
! Uncomment next line to print compilation info. BEWARE: may occasionally
! give compilation errors due to lines too long if paths are very long
!#define __HAVE_CONFIG_INFO
!
!==-----------------------------------------------------------------------==!
MODULE environment
  !==-----------------------------------------------------------------------==!

  USE kinds, ONLY: DP
  USE io_files, ONLY: crash_file, nd_nmbr
  USE io_global, ONLY: stdout, meta_ionode
  USE mp_world,  ONLY: nproc, nnode
  USE mp_images, ONLY: me_image, my_image_id, root_image, nimage, &
      nproc_image
  USE mp_pools,  ONLY: npool
  USE mp_bands,  ONLY: ntask_groups, nproc_bgrp, nbgrp, nyfft
  USE global_version, ONLY: version_number
  USE fox_init_module, ONLY: fox_init
  USE command_line_options, ONLY : nmany_
#if defined(__HDF5)
  USE qeh5_base_module,   ONLY: initialize_hdf5, finalize_hdf5
#endif

  IMPLICIT NONE

  ! ...  title of the simulation
  CHARACTER(LEN=75) :: title

  SAVE

  PRIVATE

  PUBLIC :: environment_start
  PUBLIC :: environment_end
  PUBLIC :: opening_message
  PUBLIC :: compilation_info
  PUBLIC :: parallel_info

  !==-----------------------------------------------------------------------==!
CONTAINS
  !==-----------------------------------------------------------------------==!

  SUBROUTINE environment_start( code )

    CHARACTER(LEN=*), INTENT(IN) :: code

    LOGICAL           :: exst, debug = .false.
    CHARACTER(LEN=80) :: code_version, uname
    CHARACTER(LEN=6), EXTERNAL :: int_to_char
    INTEGER :: ios, crashunit
    INTEGER, EXTERNAL :: find_free_unit

    ! ... The Intel compiler allocates a lot of stack space
    ! ... Stack limit is often small, thus causing SIGSEGV and crash
    ! ... One may use "ulimit -s unlimited" but it doesn't always work
    ! ... The following call does the same and always works
    !
#if defined(__INTEL_COMPILER)
    CALL remove_stack_limit ( )
#endif
    ! ... use ".FALSE." to disable all clocks except the total cpu time clock
    ! ... use ".TRUE."  to enable clocks

    CALL init_clocks( .TRUE. )
    CALL start_clock( TRIM(code) )

    code_version = TRIM (code) // " v." // TRIM (version_number)

    ! ... for compatibility with PWSCF
#if defined(__MPI)
    nd_nmbr = TRIM ( int_to_char( me_image+1 ))
#else
    nd_nmbr = ' '
#endif

    IF( meta_ionode ) THEN

       ! ...  search for file CRASH and delete it

       INQUIRE( FILE=TRIM(crash_file), EXIST=exst )
       IF( exst ) THEN
          crashunit = find_free_unit()
          OPEN( UNIT=crashunit, FILE=TRIM(crash_file), STATUS='OLD',IOSTAT=ios )
          IF (ios==0) THEN
             CLOSE( UNIT=crashunit, STATUS='DELETE', IOSTAT=ios )
          ELSE
             WRITE(stdout,'(5x,"Remark: CRASH file could not be deleted")')
          END IF
       END IF

    ELSE
       ! ... one processor per image (other than meta_ionode)
       ! ... or, for debugging purposes, all processors,
       ! ... open their own standard output file
!#define DEBUG
#if defined(DEBUG)
       debug = .true.
#endif
       IF (me_image == root_image .OR. debug ) THEN
          uname = 'out.' // trim(int_to_char( my_image_id )) // '_' // &
               trim(int_to_char( me_image))
          OPEN ( unit = stdout, file = TRIM(uname),status='unknown')
       ELSE
#if defined(_WIN32)
          OPEN ( unit = stdout, file='NUL:', status='unknown' )
#else
          OPEN ( unit = stdout, file='/dev/null', status='unknown' )
#endif
       END IF

    END IF
    !
    CALL opening_message( code_version )
    CALL compilation_info ( )
#if defined(__MPI)
    CALL parallel_info ( code )
#else
    CALL serial_info()
#endif
    CALL fox_init()
#if defined(__HDF5)
    CALL initialize_hdf5()
#endif
  END SUBROUTINE environment_start

  !==-----------------------------------------------------------------------==!

  SUBROUTINE environment_end( code )

    CHARACTER(LEN=*), INTENT(IN) :: code
#if defined(_HDF5)
    CALL finalize_hdf5()
#endif
    IF ( meta_ionode ) WRITE( stdout, * )

    CALL stop_clock(  TRIM(code) )
    CALL print_clock( TRIM(code) )

    CALL closing_message( )

    IF( meta_ionode ) THEN
       WRITE( stdout,'(A)')      '   JOB DONE.'
       WRITE( stdout,3335)
    END IF
3335 FORMAT('=',78('-'),'=')
    FLUSH(stdout)

    RETURN
  END SUBROUTINE environment_end

  !==-----------------------------------------------------------------------==!

  SUBROUTINE opening_message( code_version )

    CHARACTER(LEN=*), INTENT(IN) :: code_version
    CHARACTER(LEN=9)  :: cdate, ctime

    CALL date_and_tim( cdate, ctime )
    !
    WRITE( stdout, '(/5X,"Program ",A," starts on ",A9," at ",A9)' ) &
         TRIM(code_version), cdate, ctime
    !
    WRITE( stdout, '(/5X,"This program is part of the open-source Quantum ",&
         &    "ESPRESSO suite", &
         &/5X,"for quantum simulation of materials; please cite",   &
         &/9X,"""P. Giannozzi et al., J. Phys.:Condens. Matter 21 ",&
         &    "395502 (2009);", &
         &/9X,"""P. Giannozzi et al., J. Phys.:Condens. Matter 29 ",&
         &    "465901 (2017);", &
         &/9X," URL http://www.quantum-espresso.org"", ", &
         &/5X,"in publications or presentations arising from this work. More details at",&
         &/5x,"http://www.quantum-espresso.org/quote")' )
    RETURN
  END SUBROUTINE opening_message

  !==-----------------------------------------------------------------------==!

  SUBROUTINE closing_message( )

    CHARACTER(LEN=9)  :: cdate, ctime
    CHARACTER(LEN=80) :: time_str

    CALL date_and_tim( cdate, ctime )

    time_str = 'This run was terminated on:  ' // ctime // ' ' // cdate

    IF( meta_ionode ) THEN
       WRITE( stdout,*)
       WRITE( stdout,3334) time_str
       WRITE( stdout,3335)
    END IF

3334 FORMAT(3X,A60,/)
3335 FORMAT('=',78('-'),'=')

    RETURN
  END SUBROUTINE closing_message

  !==-----------------------------------------------------------------------==!
  SUBROUTINE parallel_info ( code )
    !
    CHARACTER(LEN=*), INTENT(IN) :: code
#if defined(_OPENMP)
    INTEGER, EXTERNAL :: omp_get_max_threads
    !
    WRITE( stdout, '(/5X,"Parallel version (MPI & OpenMP), running on ",&
         &I7," processor cores")' ) nproc * omp_get_max_threads()
    !
    WRITE( stdout, '(5X,"Number of MPI processes:           ",I7)' ) nproc
    !
    WRITE( stdout, '(5X,"Threads/MPI process:               ",I7)' ) &
         omp_get_max_threads()
#else
    WRITE( stdout, '(/5X,"Parallel version (MPI), running on ",&
         &I5," processors")' ) nproc 
#endif
    !
#if !defined(__GFORTRAN__) ||  ((__GNUC__>4) || ((__GNUC__==4) && (__GNUC_MINOR__>=8)))
    WRITE( stdout, '(/5X,"MPI processes distributed on ",&
         &I5," nodes")' ) nnode
#endif
    IF ( nimage > 1 ) WRITE( stdout, &
         '(5X,"path-images division:  nimage    = ",I7)' ) nimage
    IF ( npool > 1 ) WRITE( stdout, &
         '(5X,"K-points division:     npool     = ",I7)' ) npool
    IF ( nbgrp > 1 ) WRITE( stdout, &
         '(5X,"band groups division:  nbgrp     = ",I7)' ) nbgrp
    IF ( nproc_bgrp > 1 ) WRITE( stdout, &
         '(5X,"R & G space division:  proc/nbgrp/npool/nimage = ",I7)' ) nproc_bgrp
    IF ( nyfft > 1 ) WRITE( stdout, &
         '(5X,"wavefunctions fft division:  Y-proc x Z-proc = ",2I7)' ) &
         nyfft, nproc_bgrp / nyfft
    IF ( ntask_groups > 1 ) WRITE( stdout, &
         '(5X,"wavefunctions fft division:  task group distribution",/,34X,"#TG    x Z-proc = ",2I7)' ) &
         ntask_groups, nproc_bgrp / ntask_groups
    WRITE( stdout, '(5X,"Fft bands division:     nmany     = ",I7)' ) nmany_
    !
  END SUBROUTINE parallel_info

  !==-----------------------------------------------------------------------==!
  SUBROUTINE serial_info ( )
    !
#if defined(_OPENMP)
    INTEGER, EXTERNAL :: omp_get_max_threads
#endif
    !
#if defined(_OPENMP)
    WRITE( stdout, '(/5X,"Serial multi-threaded version, running on ",&
         &I4," processor cores")' ) omp_get_max_threads()
    !
#else
    WRITE( stdout, '(/5X,"Serial version")' )
#endif
    WRITE( stdout, '(5X,"Fft bands division:     nmany     = ",I7)' ) nmany_
    !
  END SUBROUTINE serial_info

  !==-----------------------------------------------------------------------==!
  SUBROUTINE compilation_info ( )
  !
  ! code borrowed by WanT - prints architecture / compilation details
  !
#if defined(__HAVE_CONFIG_INFO)
#include "configure.h"
! #include "build_date.h"
!
     !WRITE( stdout, "(2x,'        BUILT :',4x,a)" ) TRIM( ADJUSTL( &
     !__CONF_BUILD_DATE  ))
     WRITE( stdout, * ) 
     ! note: if any preprocessed variables __CONF_* exceeds 128 characters,
     ! the compilation may give error because the line exceeds 132 characters
     WRITE( stdout, "(2x,'         ARCH :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_ARCH))
     WRITE( stdout, "(2x,'           CC :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_CC))
     WRITE( stdout, "(2x,'          CPP :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_CPP))
     WRITE( stdout, "(2x,'          F90 :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_MPIF90))
     WRITE( stdout, "(2x,'          F77 :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_F77))
     WRITE( stdout, "(2x,'       DFLAGS :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_DFLAGS))
     WRITE( stdout, "(2x,'    BLAS LIBS :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_BLAS_LIBS))
     WRITE( stdout, "(2x,'  LAPACK LIBS :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_LAPACK_LIBS))
     WRITE( stdout, "(2x,'     FFT LIBS :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_FFT_LIBS))
     WRITE( stdout, "(2x,'    MASS LIBS :',4x,a)" ) TRIM( ADJUSTL( &
__CONF_MASS_LIBS))
     !
#endif
   END SUBROUTINE compilation_info

  !==-----------------------------------------------------------------------==!
END MODULE environment
!==-----------------------------------------------------------------------==!
