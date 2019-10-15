// OpenCL kernels which will run on the GPU

void kernel mat_mat_mul_gpu(const int m, const int k, const global float* A, const global float* B, global float* C) 
{
    const int rows = get_global_id(0); 
    const int cols = get_global_id(1); 
 
    //	computing one dot product per thread
    float acc = 0.0f;
    for (int i=0; i<k; i++) {
        acc += A[i*m + rows] * B[cols*k + i];
    }
 
    C[cols*m + rows] = acc;   
}

void kernel vec_scalar_gpu(global float * x, float a)
{
	x[get_global_id(0)]*=a;
}

void kernel shrink_gpu(global float * x, float threshold)
{

	int id = get_global_id(0);
    
    local float aux;
    aux = sign(x[id]) * x[id] - threshold;
    
	x[id] = sign(x[id]) * max(aux, 0.0f);
	
	if((sign(x[id]) * x[id]) < 1.175e-38)
	   x[id] = 0.0f;
	
}

void kernel mat_vec_mul_gpu(global const float * mat, global const float * vec, global float * res, local float * aux, int m,int n)
{
  
  float sum = 0.0f;
	
  //	computing a partial dot product
  for (int k=get_global_id(1);k<n;k+=get_global_size(1))
    {
      sum += mat[get_global_id(0)+m*k] * vec[k];
    }


  int rows = get_local_size(0);
  int cols = get_local_size(1); 
  int i = get_local_id(0);
  int j = get_local_id(1);  
  aux[i+rows*j] = sum;
  
  //	synchronizing all threads within the same workgroup
  barrier(CLK_LOCAL_MEM_FENCE); 

  //	performing the reduction to sum up all the partial dot products
  while ( cols > 1 )
    {
      cols = cols/2;
      if (j < cols) 
          aux[i+rows*j] += aux[i+rows*(j+cols)];
      barrier(CLK_LOCAL_MEM_FENCE); 
    }

  // 	writing the final answer
  if ( j == 0 ) 
     res[get_global_id(0)] = aux[i];
  
}

void kernel vec_sub_gpu(global float* vec1, global const float* vec2)
{
	int id = get_global_id(0);                                               
	vec1[id] = vec1[id] - vec2[id];
}

void kernel vec_add_gpu(global float* vec1, global const float* vec2)
{
	int id = get_global_id(0);                                               
	vec1[id] = vec1[id] + vec2[id];
}
