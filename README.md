# Sparse-representations-solver
  A collection of three multi-threaded and GPU accelerated algorithms written in C++ and OpenCL that are used for finding sparse solutions for large under-determined systems of linear equations.

  These solvers use the AVX extension and should run on most CPUs post-2011 (Sandy Bridge/Bulldozer and newer). In addition to that the GPU accelerated versions of these solver require an OpenCL 1.2 capable device, again most GPUs post-2011 should have support for it. 

Features :

  - solves the equation A*x=b where A is an m by n matrix with m<n and x is a vector which is sparse (has a lot of elements equal to zero)
  - three algorithms implemented :  Alternating Direction Algorithm (ADM), Fast Iterative Shrinkage-Thresholding Algorithm (FISTA),  Primal Augmented Lagrangian Method (PALM)                             
  - uses AVX SIMD extensions and is multi-threaded 
  - the majority of relevant computations are also GPU accelareted using OpenCL 
  - effective for very large systems i.e. the dimensions of dictionary A are of order 10000 x 10000 and over
  - supports 32bit and 64bit floating point formats (64bit FP GPU acceleration may be worse due to the fact that a lot of GPUs have much lower double precision throughput compared to 32bit FP)
  
  Notes :
  
    The efectiveness of each algorithm varies depepnding on the parameters fed into them but generally :
      - PALM is likely to be the most effective and requires no fidling of the parameters but is also the most expensive 
      - FISTA generally can produce worse results if the parameter lambda is unfit (smaller -> more exact solution, larger -> solution is more sparse), also FISTA is more tolerant to noisy data compared to the other two
      - ADM can fail to converge if the computed or provided singular value of A*A' is too big and may require fidling more with tau and beta (more specifically lowering tau)
      
      Performance considerations :
      - all three algorithms need the largest singular value of A*A' in order to guarantee convergence, the solvers provided can compute these, however this can be very expensive, especially for the CPU versions so if possible one should provide this value manually
      - if the data can fit into the cache of the CPU this generally means the speedup gained from the GPU solvers will be modest or you may even witness a regression in performance but if this is not the case the GPU versions should always be faster (for reference a GTX 1080 can be up to 10x times faster than an 8 core 1700X CPU)
      - the CPU version of the solvers will use all available threads by default, however the scaling may be much worse in reality, on the aforementioned 1700X going past 4 threads yields little improvement in terms of speed
      - the computations involved are mostly memory bound (i.e. matrix-vector multiplication) therefore sclaing of performance with better hardware may not be as good expected 
