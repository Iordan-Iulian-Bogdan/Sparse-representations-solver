#define TS 32    
#define WPT 8   
#define RTS (TS/WPT)

void kernel norm_dp(global double* x, global double* norm, int n)
{
    int i = 0;

    for (i = 0; i < n; i++)
    {
        norm[0] += x[i] * x[i];
    }

    norm[0] = sqrt(norm[0]);
}

void kernel norm_sp(global float* x, global float* norm, int n)
{
    int i = 0;

    for (i = 0; i < n; i++)
    {
        norm[0] += x[i] * x[i];
    }

    norm[0] = sqrt(norm[0]);
}

void kernel vec_scalar_gpu_sp(global float* x, float a)
{
    x[get_global_id(0)] *= a;
}

void kernel shrink_gpu_sp(global float* x, float threshold)
{

    const int id = get_global_id(0);

    local float aux;
    aux = sign(x[id]) * x[id] - threshold;

    x[id] = sign(x[id]) * fmax(aux, 0.0f);

    if ((sign(x[id]) * x[id]) < 1.175e-38)
        x[id] = 0.0f;

}

void kernel mat_vec_mul_gpu_sp(global const float* mat, global const float* vec, global float* res, local float* aux, int m, int n)
{

    float sum = 0.0f;

    //	computing a partial dot product
    for (int k = get_global_id(1); k < n; k += get_global_size(1))
    {
        sum += mat[get_global_id(0) + m * k] * vec[k];
    }


    const int rows = get_local_size(0);
    int cols = get_local_size(1);
    const int i = get_local_id(0);
    const int j = get_local_id(1);
    aux[i + rows * j] = sum;

    //	synchronizing all threads within the same workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    //	performing the reduction to sum up all the partial dot products
    while (cols > 1)
    {
        cols = cols / 2;
        if (j < cols)
            aux[i + rows * j] += aux[i + rows * (j + cols)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 	writing the final answer
    if (j == 0)
        res[get_global_id(0)] = aux[i];

}

void kernel vec_sub_gpu_sp(global float* vec1, global const float* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] - vec2[id];
}

void kernel vec_add_gpu_sp(global float* vec1, global const float* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] + vec2[id];
}

void kernel vec_scalar_gpu_dp(global double* x, double a)
{
    x[get_global_id(0)] *= a;
}

void kernel shrink_gpu_dp(global double* x, double threshold)
{

    const int id = get_global_id(0);

    local double aux;
    aux = sign(x[id]) * x[id] - threshold;

    x[id] = sign(x[id]) * fmax(aux, (double)(0.0f));

    if ((sign(x[id]) * x[id]) < 1.175e-38)
        x[id] = 0.0f;

}

void kernel mat_vec_mul_gpu_dp(global const double* mat, global const double* vec, global double* res, local double* aux, int m, int n)
{

    double sum = 0.0f;

    //	computing a partial dot product
    for (int k = get_global_id(1); k < n; k += get_global_size(1))
    {
        sum += mat[get_global_id(0) + m * k] * vec[k];
    }


    const int rows = get_local_size(0);
    int cols = get_local_size(1);
    const int i = get_local_id(0);
    const int j = get_local_id(1);
    aux[i + rows * j] = sum;

    //	synchronizing all threads within the same workgroup
    barrier(CLK_LOCAL_MEM_FENCE);

    //	performing the reduction to sum up all the partial dot products
    while (cols > 1)
    {
        cols = cols / 2;
        if (j < cols)
            aux[i + rows * j] += aux[i + rows * (j + cols)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 	writing the final answer
    if (j == 0)
        res[get_global_id(0)] = aux[i];

}

void kernel vec_sub_gpu_dp(global double* vec1, global const double* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] - vec2[id];
}

void kernel vec_add_gpu_dp(global double* vec1, global const double* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] + vec2[id];
}

// Increased the amount of work-per-thread by a factor WPT
__kernel void mat_mat_mul_gpu_sp(const int M, const int K,
    const __global float* A,
    const __global float* B,
    __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
            Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w = 0; w < WPT; w++) {
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}

// Increased the amount of work-per-thread by a factor WPT
__kernel void mat_mat_mul_gpu_dp(const int M, const int K,
    const __global double* A,
    const __global double* B,
    __global double* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local double Asub[TS][TS];
    __local double Bsub[TS][TS];

    // Initialise the accumulation registers
    double acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
            Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w = 0; w < WPT; w++) {
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}
