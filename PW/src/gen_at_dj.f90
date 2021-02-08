!
! Copyright (C) 2002-2020 Quantum ESPRESSO group
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .
!
!
!----------------------------------------------------------------------
SUBROUTINE gen_at_dj( ik, dwfcat )
   !----------------------------------------------------------------------
   !! This routine calculates the atomic wfc generated by the derivative
   !! (with respect to the q vector) of the bessel function. This vector
   !! is needed in computing the Hubbard contribution to the stress tensor.
   !
   USE kinds,       ONLY: DP
   USE io_global,   ONLY: stdout
   USE constants,   ONLY: tpi
   USE atom,        ONLY: msh
   USE ions_base,   ONLY: nat, ntyp => nsp, ityp, tau
   USE cell_base,   ONLY: omega, at, bg, tpiba
   USE klist,       ONLY: xk, ngk, igk_k
   USE gvect,       ONLY: mill, eigts1, eigts2, eigts3, g
   USE wvfct,       ONLY: npwx
   USE uspp_param,  ONLY: upf
   USE basis,       ONLY: natomwfc
   !
   IMPLICIT NONE
   !
   INTEGER, INTENT(IN) :: ik
   !! k-point index
   COMPLEX(DP), INTENT(OUT) :: dwfcat(npwx,natomwfc)
   !! the derivative of the atomic wfcs (all)
   !
   ! ... local variables
   !
   INTEGER :: l, na, nt, nb, iatw, iig, ig, m, lm, nwfcm, lmax_wfc, npw
   REAL(DP) :: arg
   COMPLEX(DP) :: phase, pref
   REAL(DP),    ALLOCATABLE :: gk(:,:), q(:), ylm(:,:), djl(:,:,:)
   COMPLEX(DP), ALLOCATABLE :: sk(:)
   ! 
   npw = ngk(ik)
   nwfcm = MAXVAL( upf(1:ntyp)%nwfc )
   ! calculate max angular momentum required in wavefunctions
   lmax_wfc = 0
   do nt = 1, ntyp
      lmax_wfc = MAX ( lmax_wfc, MAXVAL (upf(nt)%lchi(1:upf(nt)%nwfc) ) )
   enddo
   !
   ALLOCATE( ylm (npw,(lmax_wfc+1)**2), djl (npw,nwfcm,ntyp) )
   ALLOCATE( gk(3,npw), q (npw) )
   !
   DO ig = 1, npw
      iig = igk_k(ig,ik)
      gk(1,ig) = xk(1,ik) + g(1,iig)
      gk(2,ig) = xk(2,ik) + g(2,iig)
      gk(3,ig) = xk(3,ik) + g(3,iig)
      q(ig) = gk(1,ig)**2 + gk(2,ig)**2 + gk(3,ig)**2
   ENDDO
   !
   !  ylm = spherical harmonics
   !
   CALL ylmr2( (lmax_wfc+1)**2, npw, gk, q, ylm )
   !
   q(:) = SQRT(q(:))*tpiba
   CALL radial_dwfc ( npw, q, nwfcm, ntyp, djl )
   !
   DEALLOCATE( q, gk )
   !
   ALLOCATE( sk(npw) )
   !
   iatw = 0
   DO na=1,nat
      nt=ityp(na)
      arg = ( xk(1,ik) * tau(1,na) + &
              xk(2,ik) * tau(2,na) + &
              xk(3,ik) * tau(3,na) ) * tpi
      phase = CMPLX( COS(arg), -SIN(arg), KIND=DP )
      DO ig =1,npw
         iig = igk_k(ig,ik)
         sk(ig) = eigts1(mill(1,iig),na) *      &
                  eigts2(mill(2,iig),na) *      &
                  eigts3(mill(3,iig),na) * phase
      ENDDO
      !
      DO nb = 1,upf(nt)%nwfc
         ! Note: here we put ">=" to be consistent with "atomic_wfc"/"n_atom_wfc" 
         IF ( upf(nt)%oc(nb) >= 0.d0 ) THEN
            l = upf(nt)%lchi(nb)
            pref = (0.d0,1.d0)**l
            DO m = 1,2*l+1
               lm = l*l+m
               iatw = iatw+1
               DO ig=1,npw
                  dwfcat(ig,iatw) = djl(ig,nb,nt)*sk(ig)*ylm(ig,lm)*pref
               ENDDO
            ENDDO
         ENDIF
      ENDDO
   ENDDO
   !
   IF (iatw /= natomwfc) THEN
      WRITE( stdout,*) 'iatw =', iatw, 'natomwfc =', natomwfc
      CALL errore( 'gen_at_dj', 'unexpected error', 1 )
   ENDIF

   DEALLOCATE( sk       )
   DEALLOCATE( djl, ylm )
   !
   RETURN
   !
END SUBROUTINE gen_at_dj
