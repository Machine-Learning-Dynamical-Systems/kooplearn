C     -*- fortran -*-
C     This file is autogenerated with f2py (version:2)
C     It contains Fortran 77 wrappers to fortran functions.

      subroutine f2pywrapl_upr1fact_hess (l_upr1fact_hessf2pywrap,
     & n, p)
      external l_upr1fact_hess
      integer n
      logical p(n - 2)
      logical l_upr1fact_hessf2pywrap, l_upr1fact_hess
      l_upr1fact_hessf2pywrap = .not.(.not.l_upr1fact_hess(n, p))
      end
