dm
==

Diffusion model first passage time distribution library.

*No guarantee is provided for the correctness of the implementation.*

Content
-------

The library consists of functions written in ANSI C 89 that compute the first-passage time distribution of diffusion models. It provides various functions with different degrees of optimisation. Depending on the function, they support leaky/weighted integration, time-varying drift rates, and time-varying boundaries.

In addition to the C implementation, two MATLAB MEX function allow MATLAB users access to the library. These MEX functions call the various different library function depending on the parameters provided.

For constant drift and bounds, the functions compute the first-passage time distribution by methods described in

Cox DR and Miller HD (1965). *The Theory of Stochastic Processes*. John Wiley & Sons, Inc.

For all other cases, the implementation described in

Smith PL (2000). Stochastic Dynamic Models of Response Time and Accuracy: A Foundational Primer. In *Journal of Mathematical Psychology*, 44 (3). 408-463.

is used.

For details of the available C functions, see ddm_rt_dist_lib.h

The provided MATLAB MEX functions are

ddm_rt_dist: function for symmetric bounds. See ddm_rt_dist.m for usage information.

ddm_rt_dist_full: function for arbitrary bounds. See ddm_rt_dist_full.m for usage information.

Usage
-----

To use the C library, include ddm_rt_dist_lib.h and link to compiled ddm_rt_dist_lib.c

The MATLAB MEX functions need to be compiled before use.. To do so, run

    mex ddm_rt_dist.c ddm_rt_dist_lib.c
    mex ddm_rt_dist_full.c ddm_rt_dist_lib.c

at the command line. The location of the 'mex' executable might be OS-dependent.

Further Information
-------------------

For further information, visit the [homepage of Jan Drugowitsch](http://www.lnc.ens.fr/~jdrugowi/code_ddm.html)
