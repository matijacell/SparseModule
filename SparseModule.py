# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:03:10 2022

@author: Matija
"""
import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def dense_to_coo(A):
    '''
        Converts a dense numpy array into COO form
        
        Args: 
            A: dense numpy array
            
        Returns:
            x: nonzero values
            col: column indices
            row: row idices
            shape: shape of A
            
    '''
    
    idxs = np.argwhere(A)
    x = np.empty( shape=(len(idxs)), dtype=A.dtype ) # data, x, y

    col = idxs[:,1] # col
    row = idxs[:,0] # row

    # TODO: do this without a for loop?
    for i in range(len(idxs)):
        x[i] = A[idxs[i][0],idxs[i][1]]

    return x, col, row, np.shape(A)
    
# @TODO: datatypes fix
@jit(nopython=True)
def kronecker(Ax, A_col, A_row, A_shape, Bx, B_col, B_row, B_shape):
        '''
            Calculates Kronecker product of matrices A and B in COO form.
            
            Args: 
                Ax: nonzero values of A
                A_col: column indices of A
                A_row: row indices of A
                A_shape: shape of A
                Bx: nonzero values of B
                B_col: column indices of B
                B_row: row indices of B
                B_shape: shape of B
                
            Returns:
                data: nonzero entries
                col: column indices
                row: row indices
                output_shape: shape of output array
                
        '''
    
        output_shape = (A_shape[0]*B_shape[0], A_shape[1]*B_shape[1])

        nnz_B = len(Bx)
        
        # expand entries of a into blocks
        col = A_col.repeat(nnz_B)
        row = A_row.repeat(nnz_B)
        data = Ax.repeat(nnz_B)

        row *= B_shape[0]
        col *= B_shape[1]

        # increment block indices
        row,col = row.reshape(-1,nnz_B),col.reshape(-1,nnz_B)
        
        col += B_col
        row += B_row
        row,col = row.reshape(-1),col.reshape(-1)

        # compute block entries
        data = data.reshape(-1, nnz_B) * Bx
        data = data.reshape(-1)
        
        return data, col, row, output_shape

@jit(nopython=True)
def coo_to_array(Ax, col, row, A_shape):
    '''
        Converts a matrix in COO form back to a dense numpy array
        
        Args: 
            Ax: nonzero values
            col: column indices
            row: row indices
            A_shape: shape of matrix A

        Returns:
            C: numpy array
            
    '''
    
    C = np.zeros(shape=A_shape, dtype=Ax.dtype)
    
    # @TODO: with no for loop
    for i in range(len(Ax)):
        C[int(row[i]), int(col[i])] = Ax[i]
        
    return C

@jit(nopython=True)
def coo_to_csr(x, col, row, A_shape):
    '''
        Converts a sparse matrix in COO form to CSR
        
        Args: 
            x: nonzero values
            col: column indices
            row: row indices
            A_shape: shape of matrix A

        Returns:
            x: nonzero values
            col: column indices
            indptr: row pointers
            A_shape: shape of matrix A            
    '''
        
    indptr = np.zeros(A_shape[0]+1, dtype=numba.int16)
    
    for i in range(0, len(x) ) :
        indptr[int(row[i]) + 1] += 1
        
    for i in range(0, A_shape[0]):
        indptr[i+1] += indptr[i]
    
    return x, col, indptr, A_shape

@jit(nopython = True)
def csr_matmat(n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx, dtype):
    '''
        Compute C = A@B for CSR matrices A,B. Only updates preallocated arrays
        and return True
        
        Args: 
            n_row: number of rows in A
            n_col: number of columns in B (hence C is n_row by n_col)
            
            Ap: row pointer of A
            Aj: column indices of A
            Ax: nonzero values of A
            
            Bp: row pointers of B
            Bj: column indices of B
            Bx: nonzero values of B

            dtype: datatype of matrices A and B
            
            Cp: preallocated array of row pointers of C
            Cj: preallocated array of column indices of C
            Cx: preallocated array of nonzero values of C

        Returns:
            True
            
        Notes:
            Output arrays Cp, Cj, and Cx must be preallocated.
            In order to find the appropriate type for T, csr_matmat_maxnnz can be used
            to find nnz(C).

    '''
    
    nxt = np.empty(shape=n_col, dtype=numba.int8) # nxt = next
    nxt[:] = -1
    sums = np.zeros(shape=n_col, dtype=numba.complex64)

    nnz = 0
    Cp[0] = 0

    for i in range(0, n_row): 
        head = -2
        length = 0

        jj_start = Ap[i]
        jj_end = Ap[i+1]
        for jj in range(jj_start, jj_end):
            j = Aj[jj]
            v = Ax[jj]

            kk_start = Bp[j]
            kk_end   = Bp[j+1]
            for kk in range(kk_start, kk_end):
                k = Bj[kk]

                sums[k] += v*Bx[kk]

                if nxt[k] == -1:
                    nxt[k] = head
                    head  = k
                    length += 1

        for jj in range(0, length):
            if sums[head] != 0:
                Cj[nnz] = head
                Cx[nnz] = sums[head]
                nnz += 1
            

            temp = head
            head = nxt[head]

            nxt[temp] = -1 # clear arrays
            sums[temp] =  0
        
        Cp[i+1] = nnz;
    
    return True

# See csr matmat comments
@jit(nopython=True)
def csr_matmat_maxnnz(n_row, n_col, Ap, Aj, Bp, Bj):
    '''
        Compute nnz for preallocation of arrays when doing matrix
        multiplication in CSR form. See comments in csr_matmul and csr_matmat
        for details.
        
        Args: 
            n_row: number of rows in A
            n_col: number of columns in B (hence C is n_row by n_col)
            
            Ap: row pointer of A
            Aj: column indices of A
            
            Bp: row pointers of B
            Bj: column indices of B

        Returns:
            nnz: shape of column indices of the product of A@B

    '''
    
    
    mask = np.empty( shape= n_col ) 
    mask[:] = -1

    nnz = 0
    for i in range(0, n_row):
        row_nnz = 0;

        for jj in range(Ap[i], Ap[i+1]):
            j = Aj[jj]
            for kk in range(Bp[j], Bp[j+1]):                
                k = Bj[kk]
                if(mask[k] != i):
                    mask[k] = i
                    row_nnz += 1
                    
        nnz = nnz + row_nnz;

    return nnz

# Helper function to make code cleaner
@jit(nopython=True)
def csr_matmul(Ax, Ap, Aj, Bx, Bp, Bj, A_shape, B_shape, dtype):
    '''
        Helper function to take care of matrix multiplication
        A@B = C in CSR form.
        
        Args: 
            Ax: nonzero values of matrix A
            Ap: row pointer of matrix A
            Aj: column indices of matrix A
            Bx: nonzero values of matrix B
            Bp: row pointer of matrix B
            Bj: column indices of matrix B
            A_shape: shape of matrix A
            B_shape: shape of matrix B
            dtype: datatype of matrices A and B

        Returns:
            Cx: nonzero values of C = A@B
            Cj: column indices of C = A@B
            Cp: row pointer of C = A@B
            shape: shape of C = A@B
            

    '''
    
    n_row = A_shape[0]
    n_col = B_shape[1]
    
    nnz_max = csr_matmat_maxnnz(n_row, n_col, Ap, Aj, Bp, Bj)
        
    Cp = np.empty(shape=(n_row+1), dtype=numba.int16)
    Cj = np.empty(shape=nnz_max, dtype=numba.int16)
    Cx = np.empty_like(Cj, dtype=numba.complex64)
    
    csr_matmat(n_row, n_col, 
               Ap, Aj, Ax,
               Bp, Bj, Bx,
               Cp, Cj, Cx,
               dtype)
    
    return Cx, Cj, Cp, (n_row, n_col)
    
@jit(nopython=True)
def expandptr(n_row, Ap, Bi):
    '''
        Expand a compressed row pointer into a row array
        
        Args: 
            n_row: number of rows in A
            Ap: row pointer
            Bi: preallocated output array

        Returns:
            True
            
        Notes:
            Complexit: Linear

    '''

    for i in range(0, n_row):
        for jj in range(Ap[i], Ap[i+1]):
            Bi[jj] = i

    return True

@jit(nopython=True)
def csr_to_coo(x, col_csr, row_p, shape):
    '''
        Convert CSR matrix to COO
        
        Args: 
            x: number of rows in A
            col_csr: number of columns in A
            row_csr: row pointer
            shape: Column indices

        Returns:
            x: nonzero values
            col: column index
            row: row index
            shape: shape of matrix

    '''

    major_dim, minor_dim = shape[1], shape[0]
    minor_indices = col_csr
    major_indices = np.empty(len(minor_indices), dtype='int8')
    expandptr(major_dim, row_p, major_indices)
    col, row = minor_indices, major_indices # ne vem zakaj je tu obrnjeno lih kontra

    return  x, col, row, shape # nnz, col, row


@jit(nopython=True)
def csr_diagonal(n_row, n_col, Ap, Aj, Ax, Yx):
    '''
        Extract diagonal of CSR matrix A
        
        Args: 
            n_row: number of rows in A
            n_col: number of columns ni A
            Ap: row pointer
            Aj: Column indices
            Ax: nonzero values
            Yx: preallocated output arrray
            
        Returns:
            Yx[min(n_row,n_col)]: diagonal entries of A
            
        Notes:
            Output array Yx must be preallocated
            Duplicate entries will be summed.
            Complexity: Linear.  Specifically O(nnz(A) + min(n_row,n_col))
    '''

    first_row = 0
    first_col = 0
    N = min(n_row - first_row, n_col - first_col)

    for i in range(0, N):
        row = first_row + i
        col = first_col + i
        row_begin = Ap[row]
        row_end = Ap[row + 1]

        diag = 0
        for j in range(row_begin, row_end):
            if Aj[j] == col:
                diag += Ax[j]

        Yx[i] = diag
        
@jit(nopython=True)
def diagonal(Ax, Ap, Aj, Ashape, dtype):
    '''
        Helper function for extraction of the diagonal of a CSR matrix
        
        Args: 
            Ax: nonzero values
            Ap: row pointer
            Aj: column indices
            Ashape: shape of matrix A
            dtype: datatype of matrix A
            
        Returns:
            y: diagonal entries of A

    '''
    
    rows, cols = Ashape
    
    y = np.empty(min(rows, cols), dtype=numba.complex64)
    csr_diagonal(Ashape[0], Ashape[1], Ap, Aj, Ax, y)

    return y