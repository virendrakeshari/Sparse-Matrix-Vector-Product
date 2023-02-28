#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define MAX_NUM_WORDS 20

int *csr_row_pointer, *ellpack_row_pointer;                                                              //Array to store row pointers 
int *csr_column_indices, *ellpack_column_indices;                                                           //Array to store column indices for nonzero coefficients
double *csr_values, *ellpack_values;                                                                //Array to store coefficient values for nonzero coefficients
double *x_vector;                                                                  //Array to store random vector
double *y_serial_product_csr, *y_serial_product_ellpack;                                                      //Array to store product result of serial product computation
double *y_cuda_product_csr, *y_cuda_product_ellpack;                                                    //Array to store product result of parallel product computation
int totalRows;                                                                     //Variable to store total rows in the matrix
int totalColumns;                                                                  //Variable to store total columns in the matrix
int total_nonZeros;                                                                //Variable to store total non Zeros coefficients in the matrix
int max_nonZeros_in_any_row;                                                  //Variable to store maximum non zero coefficients in any row of the matrix
int *nonZeros_in_row;                                                              //Array to store nonzero coefficients in each row of matrix
int threads_per_block, number_of_blocks;
struct timespec parallel_start_csr, parallel_end_csr, parallel_start_ellpack, parallel_end_ellpack;

typedef struct                              //Data structure to store row indices, column indices and values for all non zero coefficients in the matrix
{                                        
    int row;
    int col;
    double val;
} sparse_matrix;

sparse_matrix *matrix_coefficient;                                                  //Defining Variable of Data structure sparse_matrix

int sort_sparsematrix(const void* a, const void* b)      //Function to sort non zero coefficients in the matrix in row-major and then column-major order
{
    sparse_matrix* elemA = (sparse_matrix*)a;
    sparse_matrix* elemB = (sparse_matrix*)b;
    //Comparing row indices for sorting row-major order
    if (elemA->row < elemB->row) 
    {
        return -1;
    }
    else if (elemA->row > elemB->row) 
    {
        return 1;
    }
    else 
    {  
        //Comparing column indices for sorting column-major order
        if (elemA->col < elemB->col) 
        {
            return -1;
        }
        else if (elemA->col > elemB->col) 
        {
            return 1;
        }
        else 
        {
            return 0;
        }
    }
}

int matrix_symmetric_pattern(char *input_filename)                      //Function to process a matrix which is symmetric as well as pattern 
{
    printf("Processing a symmetric Matrix where storage file is marked as pattern\n");
    char header[1024];
    int i,adjusted_nnz, max_possible_nnz;
    FILE *input_file = fopen(input_filename, "r");                      //Opening input file in read mode
    while (fgets(header, 1024, input_file)) 
    {
        if (header[0] == '%') 
        {
            continue;
        } 
        else 
        {
            sscanf(header, "%d %d %d", &totalRows, &totalColumns, &total_nonZeros); //Reading total rows, total columns and total non-Zero coefficients
            break;
        }
    }
    printf("\nOriginal Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Allocate memory for total nonZero coefficients, including maximum additional coefficients that could be added to complete the symmetric matrix
    max_possible_nnz= 2 * total_nonZeros;                                               // maximum number of coefficients possible in a symmetric matrix
    matrix_coefficient = (sparse_matrix*)malloc(max_possible_nnz* sizeof(sparse_matrix));
    memset(matrix_coefficient, 0, max_possible_nnz* sizeof(sparse_matrix));       
    adjusted_nnz = total_nonZeros;
    for (i = 0; i < total_nonZeros; i++) 
    {
        fgets(header, 1024, input_file);
        sscanf(header, "%d %d ", &matrix_coefficient[i].row, &matrix_coefficient[i].col);  //Reading row and column indices of nonZero coefficients
        matrix_coefficient[i].val = 1;                                                  //Assigning value of nonZero coefficients to 1
        //Converting from 1 indexed to 0 indexed
        matrix_coefficient[i].row--;
        matrix_coefficient[i].col--;
        //Creating new symmetric coefficients for all non Diagonal non Zero coefficients 
        if (matrix_coefficient[i].row != matrix_coefficient[i].col) 
        {
            sparse_matrix new_coefficient = {matrix_coefficient[i].col, matrix_coefficient[i].row, matrix_coefficient[i].val};
            matrix_coefficient[adjusted_nnz] = new_coefficient;
            adjusted_nnz++;
        } 
    }
    printf("Being pattern matrix: Assigned value = 1 for all non-Zero coefficients\n");
    printf("Being symmetric matrix: Created symmetric coefficeints for all original non-Zero coefficients\n");
    total_nonZeros = adjusted_nnz;                                                //Adjusting total number of non Zero coefficients for symmetric matrix
    printf("Reconstructed Full Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Sort the non zero coefficients in the matrix in row-major and then column-major order
    qsort(matrix_coefficient, adjusted_nnz, sizeof(sparse_matrix), sort_sparsematrix);
    printf("The nonZero coefficients are sorted in row-major and then column-major order\n");     //Confirm sorting is completed
    fclose(input_file);                                                            //Close the file
    return 0;
}

int matrix_symmetric_real(char *input_filename)                            //Function to process a matrix which is symmetric but not pattern 
{
    printf("Processing a symmetric Matrix where storage file is not marked as pattern\n");
    char header[1024];
    int i;
    int max_possible_nnz, adjusted_nnz;
    FILE *input_file = fopen(input_filename, "r");                         //Opening input file in read mode
    while (fgets(header, 1024, input_file)) 
    {
        if (header[0] == '%') 
        {
            continue;
        } 
        else 
        {
            sscanf(header, "%d %d %d", &totalRows, &totalColumns, &total_nonZeros);  //Reading total rows, total columns and total non-Zero coefficients
            break;
        }
    }
    printf("\nOriginal Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Allocate memory for total nonZero coefficients, including maximum additional coefficients that could be added to complete the symmetric matrix
    max_possible_nnz = 2 * total_nonZeros;                                           // maximum number of coefficients possible in a symmetric matrix
    matrix_coefficient = (sparse_matrix*)malloc(max_possible_nnz* sizeof(sparse_matrix));
    memset(matrix_coefficient, 0, max_possible_nnz* sizeof(sparse_matrix));                    
    adjusted_nnz = total_nonZeros;
    for (i = 0; i < total_nonZeros; i++) 
    {
        fgets(header, 1024, input_file);
        sscanf(header, "%d %d %lf", &matrix_coefficient[i].row, &matrix_coefficient[i].col, &matrix_coefficient[i].val); //Reading row and column indices and values of nonZero coefficients 
        //Converting from 1 indexed to 0 indexed
        matrix_coefficient[i].row--;
        matrix_coefficient[i].col--;
        //Creating new symmetric coefficients for all non Diagonal non Zero coefficients 
        if (matrix_coefficient[i].row != matrix_coefficient[i].col)             
        {
            sparse_matrix new_coefficient = {matrix_coefficient[i].col, matrix_coefficient[i].row, matrix_coefficient[i].val};
            matrix_coefficient[adjusted_nnz] = new_coefficient;
            adjusted_nnz++;
        } 
    }
    printf("Being symmetric matrix: Created symmetric coefficeints for all original non-Zero coefficients\n");
    total_nonZeros = adjusted_nnz;                                              //Adjusting total number of non Zero coefficients for symmetric matrix
    printf("Reconstructed Full Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Sort the non zero coefficients in the matrix in row-major and then column-major order
    qsort(matrix_coefficient, adjusted_nnz, sizeof(sparse_matrix), sort_sparsematrix);
    printf("The nonZero coefficients are sorted in row-major and then column-major order\n");     //Confirm sorting is completed
    fclose(input_file);                                                             //Close the file
    return 0;
}

int matrix_general_pattern(char *input_filename)                                     //Function to process a matrix which is not symmetric but pattern  
{
    printf("Processing a general Matrix where storage file is marked as pattern\n");
    char header[1024];
    int i;
    FILE *input_file = fopen(input_filename, "r");                                               //Opening input file in read mode
    while (fgets(header, 1024, input_file)) 
    {
        if (header[0] == '%') 
        {
            continue;
        } 
        else 
        {
            sscanf(header, "%d %d %d", &totalRows, &totalColumns, &total_nonZeros);  //Reading total rows, total columns and total non-Zero coefficients
            break;
        }
    }
    printf("\nOriginal Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Allocate memory for total nonZero coefficients
    matrix_coefficient = (sparse_matrix*)malloc(total_nonZeros * sizeof(sparse_matrix));
    memset(matrix_coefficient, 0, total_nonZeros * sizeof(sparse_matrix));
    for (i = 0; i < total_nonZeros; i++) 
    {
        fgets(header, 1024, input_file);
        sscanf(header, "%d %d %lf", &matrix_coefficient[i].row, &matrix_coefficient[i].col);    //Reading row and column indices of nonZero coefficients
        matrix_coefficient[i].val = 1;                                                          //Assigning value of nonZero coefficients to 1
        //Converting from 1 indexed to 0 indexed
        matrix_coefficient[i].row--;
        matrix_coefficient[i].col--;
    }
    printf("Being pattern matrix: Assigned value = 1 for all non-Zero coefficients\n");
    // Sort the non zero coefficients in the matrix in row-major and then column-major order
    qsort(matrix_coefficient, total_nonZeros, sizeof(sparse_matrix), sort_sparsematrix);
    printf("The nonZero coefficients are sorted in row-major and then column-major order\n");     //Confirm sorting is completed
    fclose(input_file);                                                                          //Close the file
    return 0;
}

int matrix_general_real(char *input_filename)                                  //Function to process a matrix which is not symmetric and not pattern 
{
    printf("Processing a general Matrix where storage file is not marked as pattern\n");
    char header[1024];
    int i;
    FILE *input_file = fopen(input_filename, "r");                              //Opening input file in read mode
    while (fgets(header, 1024, input_file)) 
    {
        if (header[0] == '%') 
        {
            continue;
        } 
        else 
        {
            sscanf(header, "%d %d %d", &totalRows, &totalColumns, &total_nonZeros);  //Reading total rows, total columns and total non-Zero coefficients
            break;
        }
    }
    printf("\nOriginal Matrix -> Rows : %d Column %d Non zeros %d\n", totalRows, totalColumns, total_nonZeros);
    // Allocate memory for total nonZero coefficients
    matrix_coefficient = (sparse_matrix*)malloc(total_nonZeros * sizeof(sparse_matrix));
    memset(matrix_coefficient, 0, total_nonZeros * sizeof(sparse_matrix));
    for (i = 0; i < total_nonZeros; i++) 
    {
        fgets(header, 1024, input_file);
        sscanf(header, "%d %d %lf", &matrix_coefficient[i].row, &matrix_coefficient[i].col, &matrix_coefficient[i].val);  //Reading row and column indices and values of nonZero coefficients 
        //Converting from 1 indexed to 0 indexed
        matrix_coefficient[i].row--;
        matrix_coefficient[i].col--; 
    }
    // Sort the non zero coefficients in the matrix in row-major and then column-major order
    qsort(matrix_coefficient, total_nonZeros, sizeof(sparse_matrix), sort_sparsematrix);
    printf("The nonZero coefficients are sorted in row-major and then column-major order\n");     //Confirm sorting is completed
    fclose(input_file);                                                                 //Close the file
    return 0;
}

int readinputfile(char *input_filename)                                                 //function to read input file
{
    char header[1024];
    int i;
    FILE *input_file = fopen(input_filename, "r");                                      //Opening input file in read mode
    char words[MAX_NUM_WORDS][1024];                                                    //Array for storing words in file banner
    int num_words = 0;
    
    printf("\nReading input file %s\n", input_filename);
    
    fgets(header, 1024, input_file);                                                    // Read the first line of the file
    char* token = strtok(header, " \t\n");                                              // Extract words from the line
    //Copy extracted words in the array
    while (token != NULL && num_words < MAX_NUM_WORDS)                         
    {
        strcpy(words[num_words], token);
        num_words++;
        token = strtok(NULL, " \t\n");
    }
    
    int is_symmetric = 0;
    int is_pattern = 0;
    // search for "symmetric" and "pattern" in the header
    for (i = 0; i < MAX_NUM_WORDS; i++) 
    {
        if (strcmp(words[i], "symmetric") == 0) 
        {
            is_symmetric = 1;
        } 
        else if (strcmp(words[i], "pattern") == 0) 
        {
            is_pattern = 1;
        }
    }

    if (is_symmetric && is_pattern) 
    {
        printf("Matrix is symmetric and pattern\n");                        //Call function to deal with symmetric and pattern matrix
        matrix_symmetric_pattern(input_filename);
    } 
    else if (is_symmetric) 
    {
        printf("Matrix is symmetric but not pattern\n");                    //Call function to deal with symmetric but not pattern matrix
        matrix_symmetric_real(input_filename);
    } 
    else if (is_pattern) 
    {
        printf("Matrix is not symmetric but pattern\n");                    //Call function to deal with non-symmetric and pattern matrix
        matrix_general_pattern(input_filename);
    }
    else 
    {
        printf("Matrix is not symmetric and not pattern\n");                //Call function to deal with non-symmetric and not pattern matrix
        matrix_general_real(input_filename);
    }
    fclose(input_file);                                                     //Close the file
    return 0;
}

void sparsetoCSR()                                                          //Function to convert sparse matrix to CSR storage format
{
    int i;
    printf("\nConverting Sparse Matrix to CSR Storage format\n");
    //Allocating memory for CSR arrays
    csr_row_pointer = (int*) malloc((totalRows + 1) * sizeof(int));          //Allocating memory for row pointer array
    csr_column_indices = (int*) malloc(total_nonZeros * sizeof(int));        //Allocating memory for column indices array
    csr_values = (double*) malloc(total_nonZeros * sizeof(double));          //Allocating memory for coefficient values array
    //Initialising row pointer array
    for (i = 0; i <= totalRows; i++)
    {
        csr_row_pointer[i] = 0;
    }
    // Read all non-zero coefficients to fill CSR arrays
    for (i = 0; i < total_nonZeros; i++)
    {
        csr_values[i] = matrix_coefficient[i].val;                                      //Filling coefficient values array
        csr_column_indices[i] = matrix_coefficient[i].col;                              //Filling column indices array
        csr_row_pointer[matrix_coefficient[i].row + 1]++;                               //Increment the count of non-zero coefficients in the current row
     }  
    //Convert the row pointer array into a cumulative sum array
    for (i = 1; i <= totalRows; i++)
    {
        csr_row_pointer[i] += csr_row_pointer[i - 1];
    } 
    printf("\nMatrix converted into CSR Storage format\n");
}

void print_CSR()                                                    //Function to print dimensions and arrays for matrix stored in CSR storage format
{
    int i;
    printf("Results for CSR:\n");
    printf("Rows (M): %d\n", totalRows);
    printf("Columns (N): %d\n", totalColumns);
    printf("Non-zeros: %d\n", total_nonZeros);
    //Print row pointers array
    printf("Row pointers (IRP): ");
    for (i = 0; i <= totalRows; i++)
        printf("%d ", csr_row_pointer[i]);
    printf("\n");
    //Print column indices array
    printf("Column indices (JA): ");
    for (i = 0; i < total_nonZeros; i++)
        printf("%d ", csr_column_indices[i]);
    printf("\n");
    //Print coefficients values array
    printf("Coefficients Values (AS): ");
    for (i = 0; i < total_nonZeros; i++)
        printf("%lf ", csr_values[i]);
    printf("\n");
}

void generate_random_vector()                                                       //Function to generate random vector
{
    int i;
    srand(9876);
    x_vector = (double*)malloc(totalColumns * sizeof(double));                      //Allocating memory for random vector
    for (i = 0; i < totalColumns; i++) 
    {
        x_vector[i] = (double)rand() / RAND_MAX * 99 + 1;        //Generates random floating-point number between 1 and 100 and assigns it to the i-th element of the x_vector array.
    }
    printf("\nRandom vector generated\n");
}

void matrix_vector_product_CSR()
{    
    int i,j;
    // MAtrix Vector Product
    y_serial_product_csr = (double*)malloc(totalRows * sizeof(double));              //Allocating memory for array to store the serial product result
    // Undertaking Sparse matrix Vector Product
    for (i = 0; i < totalRows; i++) 
    {
        double t = 0.0;
        for (j = csr_row_pointer[i]; j < csr_row_pointer[i+1]; ++j)
        {   
            t += csr_values[j] * x_vector[csr_column_indices[j]]; 
        }
        y_serial_product_csr[i] = t;
    }
}

__global__ void matrix_vector_product_CSR_kernel(double* values, int* col_indices, int* row_pointers, double* x_vector, double* y, int totalRows, int totalColumns)
{
    int i,j;
    i = blockDim.x* blockIdx.x + threadIdx.x;
    if (i < totalRows)
    {
        double t = 0.0;
        for (j = row_pointers[i]; j < row_pointers[i+1]; ++j)
        {
            t += values[j] * x_vector[col_indices[j]];
        }
        y[i] = t;
        //printf("Thread %d handling row %d\n", i, blockIdx.x * blockDim.x + threadIdx.x);
    }
}

void matrix_vector_product_CSR_cuda(double *y_cuda_product_csr)
{
    // Allocate device memory
    int i;
    double* device_csr_values;
    int* device_csr_column_indices;
    int* device_csr_row_pointer;
    double* device_x;
    double* device_y_cuda_product;

    cudaMalloc(&device_csr_values, (csr_row_pointer[totalRows]) * sizeof(double));
    cudaMalloc(&device_csr_column_indices, (csr_row_pointer[totalRows]) * sizeof(int));
    cudaMalloc(&device_csr_row_pointer, (totalRows + 1) * sizeof(int));
    cudaMalloc(&device_x, totalColumns * sizeof(double));
    cudaMalloc(&device_y_cuda_product, totalRows * sizeof(double));
    
    // Copy data to device memory
    cudaMemcpy(device_csr_values, csr_values, (csr_row_pointer[totalRows]) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_csr_column_indices, csr_column_indices, (csr_row_pointer[totalRows]) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_csr_row_pointer, csr_row_pointer, (totalRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x_vector, totalColumns * sizeof(double), cudaMemcpyHostToDevice);
    
    // Define kernel configuration
    threads_per_block = 256;
    number_of_blocks = (totalRows + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    clock_gettime(CLOCK_MONOTONIC, &parallel_start_csr);                //Start timer for calculating time for product 
    matrix_vector_product_CSR_kernel<<<number_of_blocks, threads_per_block>>>(device_csr_values, device_csr_column_indices, device_csr_row_pointer, device_x, device_y_cuda_product, totalRows, totalColumns);
    clock_gettime(CLOCK_MONOTONIC, &parallel_end_csr);                  //End timer for calculating time for product 
    
    // Copy output data from device
    cudaMemcpy(y_cuda_product_csr, device_y_cuda_product, totalRows * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_csr_values);
    cudaFree(device_csr_column_indices);
    cudaFree(device_csr_row_pointer);
    cudaFree(device_x);
    cudaFree(device_y_cuda_product);
}

void product_results_csr()                                                        //Function to print serial and parallel matrix vector product results
{
    int i;
    //Printing results for serial product computation
    printf("The final vector product for CSR in serial is [\n");
    for (int i = 0; i < totalRows; ++i) 
    {
        printf("%.6f  ", y_serial_product_csr[i]);
    }
    printf("]\n");
    //Printing results for parallel product computation
    printf("The final vector product for CSR in parallel is [\n");
    for (i = 0; i < totalRows; ++i) 
    {
        printf("%.6f  ", y_cuda_product_csr[i]);
    }
     printf("]\n");
}

void compare_results_csr(double tolerance)
{
    int count = 0;
    printf("\nComparing the results of serial and parallel Sparse Matrix vector product in CSR storage format\n");
    for (int i = 0; i < totalColumns; i++) 
    {
        double diff = fabs(y_serial_product_csr[i] - y_cuda_product_csr[i]);
        if (diff > tolerance) 
        {
            count ++;
            printf("Arrays differ at index %d: y_serial_product_csr[%d] = %.6f, y_cuda_product_csr[%d] = %.6f, diff = %.6f\n", i, i, y_serial_product_csr[i], i, y_cuda_product_csr[i], diff);
        }
    }
    if (count == 0) {
        printf("Arrays are identical\n");
    }
}

void calculate_max_nonZeros_in_any_row()                                     //Function to calculate Maximum non-Zeros in any row
{
    // Find max number of non-zeros in any row
    int i;
    max_nonZeros_in_any_row = 0;
    int curr_row = -1;
    int nonZeros_in_current_row = 0;
    for (i = 0; i < total_nonZeros; i++) 
    {
        if (matrix_coefficient[i].row != curr_row) 
        {
            if (nonZeros_in_current_row > max_nonZeros_in_any_row) 
            {
                max_nonZeros_in_any_row = nonZeros_in_current_row;
            }
            curr_row = matrix_coefficient[i].row;
            nonZeros_in_current_row = 1;
        } 
        else 
        {
        nonZeros_in_current_row++;
        }
    }
    if (i = total_nonZeros -1 && nonZeros_in_current_row > max_nonZeros_in_any_row) 
    {
        max_nonZeros_in_any_row = nonZeros_in_current_row;
    }
    printf("Max non zeros in a row : %d\n", max_nonZeros_in_any_row);
}

void sparsetoEllpack()                                                      //Function to convert sparse matrix to Ellpack storage format
{
    int i;
    printf("\nConverting Sparse Matrix to Ellpack Storage format\n");
    calculate_max_nonZeros_in_any_row();                                    //Calling function to calculate Maximum non-Zeros in any row
    //Define dimensions of padded matrix
    int num_cols_padded = max_nonZeros_in_any_row;
    int num_rows_padded = totalRows;
    int padded_size = num_cols_padded * num_rows_padded;
    
    //Allocating memory for Ellpack arrays
    ellpack_row_pointer = (int *) malloc((num_rows_padded + 1) * sizeof(int));  //Allocating memory for row pointer array
    ellpack_column_indices = (int *) malloc(padded_size * sizeof(int));         //Allocating memory for column indices array
    ellpack_values = (double *) malloc(padded_size * sizeof(double));           //Allocating memory for coefficient values array
    nonZeros_in_row = (int *) calloc(num_rows_padded, sizeof(int));             //Allocating memory for array storing non Zeros in all rows
    
    ellpack_row_pointer[0] = 0;                                                  //Initialising row pointer array
    
    // Fill row_ptr and nonZeros_in_row arrays
    int current_row = 0;
    int current_row_start = 0;
    
    for (i = 0; i < total_nonZeros; i++) 
    {
        if (matrix_coefficient[i].row == current_row) 
        {
            nonZeros_in_row[current_row]++;
        } 
        else 
        {
            current_row++;
            current_row_start = i;
            nonZeros_in_row[current_row]++;
        }
        //Filling column indices array
        ellpack_column_indices[current_row * num_cols_padded + nonZeros_in_row[current_row]-1] = matrix_coefficient[i].col;
        //Filling coefficient values array
        ellpack_values[current_row * num_cols_padded + nonZeros_in_row[current_row]-1] = matrix_coefficient[i].val;
    }
    // Filling row pointer array
    for (i = 0; i < totalRows; i++) 
    {
        ellpack_row_pointer[i+1] = ellpack_row_pointer[i] + max_nonZeros_in_any_row;
    }
    
    // Pad column indices and values arrays with 0 where non-Zeros in a particular row is less than maximum non-Zeros in any row
    for (i = 0; i < totalRows; i++) 
    {
        while (nonZeros_in_row[i] < max_nonZeros_in_any_row) 
        {
            ellpack_column_indices[i*num_cols_padded + nonZeros_in_row[i]] = 0;
            ellpack_values[i*num_cols_padded + nonZeros_in_row[i]] = 0.0;
            nonZeros_in_row[i]++;
        }
    }
    printf("\nMatrix converted into Ellpack Storage format\n");
}

void print_ellpack()                                  //Function to print dimensions and arrays for matrix stored in Ellpack storage format
{
    int i,j;
    printf("Rows (M): %d\n", totalRows);
    printf("Columns (N): %d\n", totalColumns);
    printf("Non-zeros (total_nonZeros): %d\n", total_nonZeros);
    printf("Maximum non zeroes in a row (MAXNZ): %d\n", max_nonZeros_in_any_row);
    // Print row pointers array
    printf("Row Pointers (IRP):\n");
    for (i = 0; i < totalRows; i++) 
    {
    printf("%d ", ellpack_row_pointer[i]);
    }
    printf("\n");
    // Print column indices array
    printf("Column Indices (JA):\n");
    for (i = 0; i < totalRows; i++) 
    {
        int num_nz = nonZeros_in_row[i];
        for (j = 0; j < max_nonZeros_in_any_row; j++) 
        {
            if (j < num_nz) 
            {
            printf("%d ", ellpack_column_indices[i * max_nonZeros_in_any_row + j]);
            } 
            else 
            {
            printf("0 ");
            }
        }
    printf("\n");
    }
    //Print coefficients values array
    printf("Coefficients Values (AS):\n");
    for (i = 0; i < totalRows; i++) 
    {
        int num_nz = nonZeros_in_row[i];
        for (j = 0; j < max_nonZeros_in_any_row; j++) 
        {
            if (j < num_nz) 
            {
                printf("%f ", ellpack_values[i * max_nonZeros_in_any_row + j]);
            } 
            else 
            {
            printf("0 ");
            }
        }
    printf("\n");
    }
}

void matrix_vector_product_ellpack()                                                 //Function to compute serial matrix vector product
{  
    int i, j;
    double t;
    y_serial_product_ellpack = (double*)malloc(totalRows * sizeof(double));                  //Allocating memory for array to store the serial product result
    // Undertaking Sparse matrix Vector Product
    for (i = 0; i < totalRows; i++) 
    {
        t = 0.0;
        for (j = 0; j < max_nonZeros_in_any_row; j++) 
        {
            t += ellpack_values[i * max_nonZeros_in_any_row + j] * x_vector[ellpack_column_indices[i * max_nonZeros_in_any_row + j]];                   
        }
        y_serial_product_ellpack[i] = t;
    }
}

__global__ void matrix_vector_product_ellpack_kernel(double* values, int* col_indices, int* row_lengths, int totalRows, int totalColumns, int max_nonZeros_in_any_row, double* x_vector, double* y)
{
    int i,j;
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < totalRows)
    {
        double t = 0.0;
        for (j = 0; j < max_nonZeros_in_any_row; j++) 
        {
            t += values[i * max_nonZeros_in_any_row + j] * x_vector[col_indices[i * max_nonZeros_in_any_row + j]];
        }
        y[i] = t;
        //printf("Thread %d handling row %d\n", i, blockIdx.x * blockDim.x + threadIdx.x);
    }
}

void matrix_vector_product_ellpack_cuda(double *y_cuda_product_ellpack)
{
    // Allocate device memory
    int i;
    double* device_ellpack_values;
    int* device_ellpack_col_ind;
    int* device_nonZeros_in_row;
    double* device_x;
    double* device_y_cuda_product;

    cudaMalloc(&device_ellpack_values, totalRows * max_nonZeros_in_any_row * sizeof(double));
    cudaMalloc(&device_ellpack_col_ind, totalRows * max_nonZeros_in_any_row * sizeof(int));
    cudaMalloc(&device_nonZeros_in_row, totalRows  * sizeof(int));
    cudaMalloc(&device_x, totalColumns * sizeof(double));
    cudaMalloc(&device_y_cuda_product, totalRows * sizeof(double));
    
    // Copy data to device memory
    cudaMemcpy(device_ellpack_values, ellpack_values,  totalRows * max_nonZeros_in_any_row * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ellpack_col_ind, ellpack_column_indices, totalRows * max_nonZeros_in_any_row * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_nonZeros_in_row, nonZeros_in_row, totalRows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x_vector, totalColumns * sizeof(double), cudaMemcpyHostToDevice);
    
    // Define kernel configuration
    int threads_per_block = 256;
    int number_of_blocks = (totalRows + threads_per_block - 1) / threads_per_block;
    //printf("Number of blocks = %d   Threads per Block = %d\n", number_of_blocks, threads_per_block);
    
    // Launch kernel
    clock_gettime(CLOCK_MONOTONIC, &parallel_start_ellpack);
    matrix_vector_product_ellpack_kernel<<<number_of_blocks, threads_per_block>>>(device_ellpack_values, device_ellpack_col_ind, device_nonZeros_in_row, totalRows, totalColumns, max_nonZeros_in_any_row, device_x, device_y_cuda_product);
    clock_gettime(CLOCK_MONOTONIC, &parallel_end_ellpack);
    
    // Copy output data from device
    cudaMemcpy(y_cuda_product_ellpack, device_y_cuda_product, totalRows * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_ellpack_values);
    cudaFree(device_ellpack_col_ind);
    cudaFree(device_nonZeros_in_row);
    cudaFree(device_x);
    cudaFree(device_y_cuda_product);
}

void product_results_ellpack()                                                        //Function to print serial and parallel matrix vector product results
{
    int i;
    //Printing results for serial product computation
    printf("The final vector product for Ellpack in serial is [\n");
    for (int i = 0; i < totalRows; ++i) 
    {
        printf("%.6f  ", y_serial_product_ellpack[i]);
    }
    printf("]\n");
    //Printing results for parallel product computation
    printf("The final vector product for Ellpack in parallel is [\n");
    for (i = 0; i < totalRows; ++i) 
    {
        printf("%.6f  ", y_cuda_product_ellpack[i]);
    }
     printf("]\n");
}



void compare_results_ellpack(double tolerance)
{
    int count = 0;
    printf("\nComparing the results of serial and parallel Sparse Matrix vector product\n");
    for (int i = 0; i < totalColumns; i++) 
    {
        double diff = fabs(y_serial_product_ellpack[i] - y_cuda_product_ellpack[i]);
        if (diff > tolerance) 
        {
            count ++;
            printf("Arrays differ at index %d: y_serial_product_ellpack[%d] = %.6f, y_cuda_product_ellpack[%d] = %.6f, diff = %.6f\n", i, i, y_serial_product_ellpack[i], i, y_cuda_product_ellpack[i], diff);
        }
    }
    if (count == 0) {
        printf("Arrays are identical\n");
    }
}

void compare_results_ellpack_csr(double tolerance)
{
    int count = 0;
    printf("\nComparing the results of Sparse Matrix vector product for CSR and Ellpack formats\n");
    for (int i = 0; i < totalColumns; i++) 
    {
        double diff = fabs(y_serial_product_ellpack[i] - y_serial_product_csr[i]);
        if (diff > tolerance) 
        {
            count ++;
            printf("Arrays differ at index %d: y_serial_product_ellpack[%d] = %.6f, y_serial_product_csr[%d] = %.6f, diff = %.6f\n", i, i, y_serial_product_ellpack[i], i, y_serial_product_csr[i], diff);
        }
    }
    if (count == 0) {
        printf("Arrays are identical\n");
    }
}


int main(int argc, char **argv)
{
    struct timespec start_time, end_time, serial_start_csr, serial_end_csr, serial_start_ellpack, serial_end_ellpack;
    long time_ns, serial_time_ns_csr, parallel_time_ns_csr, serial_time_ns_ellpack, parallel_time_ns_ellpack;
    double time_sec, average_serial_time_csr = 0.0, average_parallel_time_csr = 0.0, serial_time_sec_csr, parallel_time_sec_csr, T_serial_csr, T_parallel_csr, FLOPS_serial_csr, FLOPS_parallel_csr, serial_time_sec_ellpack, average_serial_time_ellpack = 0.0, average_parallel_time_ellpack = 0.0, parallel_time_sec_ellpack, T_serial_ellpack, T_parallel_ellpack, FLOPS_serial_ellpack, FLOPS_parallel_ellpack;
    int num_iterations = 100;
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);                                  //Starting program time
    
    readinputfile(argv[1]);                                                       //Calling function to read input file
    
    sparsetoCSR();                                                                //Calling function to convert sparse matrix to CSR storage format
    //print_CSR();                                                                //Calling function to print dimensions and arrays for the matrix
    
    generate_random_vector();                                                     //Calling function to generate random vector

    //Undertake serial matrix vector product
    printf("\nUndertaking serial Sparse matrix vector Product for CSR storage:\t");
    printf("\nResults for serial computation \n");
    for (int i = 0; i < num_iterations; i++) 
    {
        clock_gettime(CLOCK_MONOTONIC, &serial_start_csr);                          // start timer
        matrix_vector_product_CSR();                                                // call function for serial product
        clock_gettime(CLOCK_MONOTONIC, &serial_end_csr);                            // stop timer
        serial_time_ns_csr = (serial_end_csr.tv_sec - serial_start_csr.tv_sec) * 1000000000L        //Calculating serial product time in nanoseconds
                    + serial_end_csr.tv_nsec - serial_start_csr.tv_nsec;
        serial_time_sec_csr = (double)serial_time_ns_csr / 1000000000.0;                    //Converting serial product time in seconds
        //printf("Time for iteration %d: %.9f seconds \n", i+1, serial_time_sec_csr);
        average_serial_time_csr += serial_time_sec_csr; // accumulate total time
    }
    average_serial_time_csr /= num_iterations; // compute average time
    printf("Average time for serial product computation after %d iterations : %.9f seconds \n ", num_iterations, average_serial_time_csr);
    T_serial_csr = average_serial_time_csr;                        
    FLOPS_serial_csr = ((2*total_nonZeros)/T_serial_csr);                                   //Computing performance in FLOPS
    printf("Serial product Performance : %.15g GFLOPS\n", (FLOPS_serial_csr/1000000000));   //Computing performance in GFLOPS
    
    //Undertake parallel matrix vector product
    y_cuda_product_csr = (double*)malloc(totalRows * sizeof(double));
    printf("\nUndertaking parallel Sparse matrix vector Product for CSR storage using CUDA\n");
    printf("\nResults for parallel computation \n");
    for (int i = 0; i < num_iterations; i++) 
    {
        matrix_vector_product_CSR_cuda(y_cuda_product_csr); // call host function for parallel product computation
        parallel_time_ns_csr = (parallel_end_csr.tv_sec - parallel_start_csr.tv_sec) * 1000000000L        //Calculating serial product time in nanoseconds
                    + parallel_end_csr.tv_nsec - parallel_start_csr.tv_nsec;
        parallel_time_sec_csr = (double)parallel_time_ns_csr / 1000000000.0;
        //printf("Time for iteration %d: %.9f seconds \n", i+1, parallel_time_sec_csr);
        average_parallel_time_csr += parallel_time_sec_csr; // accumulate total time
    }
    average_parallel_time_csr /= num_iterations; // compute average time
    printf("Number of blocks = %d   Threads per Block = %d\n", number_of_blocks, threads_per_block);
    printf("Average time for parallel product computation after %d iterations : %.9f seconds \n ", num_iterations, average_parallel_time_csr);
    T_parallel_csr = (average_parallel_time_csr);                        
    FLOPS_parallel_csr = ((2*total_nonZeros)/T_parallel_csr);                                   //Computing performance in FLOPS
    printf("Parallel product Performance : %.15g GFLOPS\n", (FLOPS_parallel_csr/1000000000));   //Computing performance in GFLOPS
    
    //product_results_csr();                                               //Calling function to print results for serial and parallel product computation
    
    compare_results_csr(1000);                                             //Calling function to compare results for serial and parallel product computation
    
    sparsetoEllpack();                                                            //Calling function to convert sparse matrix to Ellpack storage format
    //print_ellpack();                                                            //Calling function to print dimensions and arrays for the matrix
    
    //Undertake serial matrix vector product
    printf("\nUndertaking serial Sparse matrix vector Product for Ellpack storage:\t");
    printf("\nResults for serial computation \n");
    for (int i = 0; i < num_iterations; i++) 
    {
        clock_gettime(CLOCK_MONOTONIC, &serial_start_ellpack);                                // start timer
        matrix_vector_product_ellpack();                                                        // call function for serial product
        clock_gettime(CLOCK_MONOTONIC, &serial_end_ellpack); // stop timer
        serial_time_ns_ellpack = (serial_end_ellpack.tv_sec - serial_start_ellpack.tv_sec) * 1000000000L        //Calculating serial product time in nanoseconds
                    + serial_end_ellpack.tv_nsec - serial_start_ellpack.tv_nsec;
        serial_time_sec_ellpack = (double)serial_time_ns_ellpack / 1000000000.0;
        //printf("Time for iteration %d: %.9f seconds \n", i+1, serial_time_sec_ellpack);
        average_serial_time_ellpack += serial_time_sec_ellpack;                             // accumulate total time
    }
    average_serial_time_ellpack /= num_iterations;                          // compute average time
    printf("Average time for serial product computation after %d iterations : %.9f seconds \n ", num_iterations, average_serial_time_ellpack);
    T_serial_ellpack = (average_serial_time_ellpack);                        
    FLOPS_serial_ellpack = ((2*total_nonZeros)/T_serial_ellpack);                                   //Computing performance in FLOPS
    printf("Serial product Performance : %.15g GFLOPS\n", (FLOPS_serial_ellpack/1000000000));
    
    //Undertake parallel matrix vector product
    y_cuda_product_ellpack = (double*)malloc(totalRows * sizeof(double));
    printf("\nUndertaking parallel Sparse matrix vector Product for ellpack storage using CUDA\n");
    printf("\nResults for parallel computation \n");
    for (int i = 0; i < num_iterations; i++) 
    {
        matrix_vector_product_ellpack_cuda(y_cuda_product_ellpack); // call host function for parallel product computation
        parallel_time_ns_ellpack = (parallel_end_ellpack.tv_sec - parallel_start_ellpack.tv_sec) * 1000000000L        //Calculating serial product time in nanoseconds
                    + parallel_end_ellpack.tv_nsec - parallel_start_ellpack.tv_nsec;
        parallel_time_sec_ellpack = (double)parallel_time_ns_ellpack / 1000000000.0;
        //printf("Time for iteration %d: %.9f seconds \n", i+1, parallel_time_sec_ellpack);
        average_parallel_time_ellpack += parallel_time_sec_ellpack; // accumulate total time
    }
    average_parallel_time_ellpack /= num_iterations; // compute average time
    printf("Number of blocks = %d   Threads per Block = %d\n", number_of_blocks, threads_per_block);
    printf("Average time for parallel product computation after %d iterations : %.9f seconds \n ", num_iterations, average_parallel_time_ellpack);
    T_parallel_ellpack = (average_parallel_time_ellpack);                        
    FLOPS_parallel_ellpack = ((2*total_nonZeros)/T_parallel_ellpack);                                   //Computing performance in FLOPS
    printf("Parallel product Performance : %.15g GFLOPS\n", (FLOPS_parallel_ellpack/1000000000));
    
    //product_results_ellpack();                                               //Calling function to print results for serial and parallel product computation
    
    compare_results_ellpack(1000);                                          //Calling function to compare results of serial and CUDA product results 
    
    compare_results_ellpack_csr(1000);                                      //Calling function to compare results of CSR and Ellpack products
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);                                    //Ending program time
    
    //Calculating program time
    time_ns = (end_time.tv_sec - start_time.tv_sec) * 1000000000L                    //Calculating program time in nanoseconds
                    + end_time.tv_nsec - start_time.tv_nsec;      
    time_sec = (double)time_ns / 1000000000.0;                                       //Converting program time from nanoseconds to seconds
    printf("\nTotal program time: %.9f seconds\n\n", time_sec);
    return 0;
}

