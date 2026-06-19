#include "../include/tenCor.h"
#include <stdlib.h>

void matrixMultiplyCore(tensor *result, const double *data1, const double *data2, int M, int K, int N, int batch, bool trans1, bool trans2);
static void gaussianEliminationCore(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign);
static bool gaussJordanCore(tensor *augMat, int matSize, int augCols, int entryIdx);

tensor *tensorTranspose(const tensor *ten, const int *axes)
{
    int *resultShape = malloc(ten->dimensions * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }

    // reshape the tensor
    for (int i = 0; i < ten->dimensions; ++i)
    {
        resultShape[i] = ten->shape[axes[i]];
    }

    tensor *result = createTensor(ten->dimensions, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to malloc for result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    int *newCoords = malloc(ten->dimensions * sizeof(int));
    if (newCoords == NULL)
    {
        printf("Error: Failed to malloc for newCoords!\n");
        destroyTensor(result);
        return NULL;
    }

    int *oldCoords = malloc(ten->dimensions * sizeof(int));
    if (oldCoords == NULL)
    {
        printf("Error: Failed to malloc for oldCoords!\n");
        free(newCoords);
        destroyTensor(result);
        return NULL;
    }

    for (int i = 0; i < result->size; ++i)
    {
        // locate addresses in the new tensor
        // row-major
        int temp = i;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            newCoords[d] = temp % result->shape[d]; // jump this number of steps in dimension d
            temp /= result->shape[d]; // accumulate jumps at dimension d and bring it onto dimension d - 1
        }

        // maps back to the address in the old tensor
        for (int d = 0; d < result->dimensions; ++d)
        {
            oldCoords[axes[d]] = newCoords[d];
        }

        int oldI = 0;
        int multiplier = 1;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            oldI += oldCoords[d] * multiplier; // accumulate jumps
            multiplier *= ten->shape[d]; // magnitude of a jump at the next dimension, so-called "weight" of each dimension
        }

        // finally assign to the new tensor's data
        result->data[i] = ten->data[oldI];
    }

    free(newCoords);
    free(oldCoords);
    return result;
}

tensor *tensorMultiply(const tensor *ten1, const tensor *ten2, bool trans1, bool trans2)
{
    // ten1->shape: {10, 5, 3, 4}
    // to be multiplicable: ten2->shape: {10, 5, 4, int N}
    // whereas: {10, 5}: batched dimensions

    // validation
    if (ten1->dimensions < 2 || ten2->dimensions < 2)
    {
        printf("Error: Tensors' ranks must be at least 2!\n");
        return NULL;
    }
    else if (ten1->dimensions != ten2->dimensions)
    {
        printf("Error: Tensors' dimensions must be the same!\n");
        return NULL;
    }
    int dims = ten1->dimensions;

    // check batch dimensions (0..(dimensions - 3))
    for (int d = 0; d < dims - 2; ++d)
    {
        if (ten1->shape[d] != ten2->shape[d])
        {
            printf("Error: Dissimilar batch dimension!\nten1->shape[%i] != ten2->shape[%i]!\n", d, d);
            return NULL;
        }
    }

    // obtain the actual dimensionality of the will-be-multiplied matrices
    int firstRows  = (trans1) ? ten1->shape[dims - 1] : ten1->shape[dims - 2];
    int firstCols  = (trans1) ? ten1->shape[dims - 2] : ten1->shape[dims - 1];
    
    int secondRows = (trans2) ? ten2->shape[dims - 1] : ten2->shape[dims - 2];
    int secondCols = (trans2) ? ten2->shape[dims - 2] : ten2->shape[dims - 1];
    if (secondRows != firstCols)
    {
        printf("Error: Matrices not multiplicable! (%dx%d and %dx%d)\n", firstRows, firstCols, secondRows, secondCols);
        return NULL;
    }
    // resultShape
    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    for (int i = 0; i < dims - 2; ++i)
    {
        resultShape[i] = ten1->shape[i];
    }
    resultShape[dims - 2] = firstRows;
    resultShape[dims - 1] = secondCols;
    
    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    tensorFillZEROS(result);

    // total numbers of matrices to be multiplied
    int totBatches = 1;
    for (int i = 0; i < dims - 2; ++i)
    {
        totBatches *= ten1->shape[i];
    }

    // multiply each pair of matrices
    for (int b = 0; b < totBatches; ++b)
    {
        matrixMultiplyCore(result, ten1->data, ten2->data, firstRows, firstCols, secondCols, b, trans1, trans2);
    }
    return result;
}

double matrixDet(const tensor *mat)
{
    // check if the inputs is an actual matrix
    int dim = mat->dimensions;
    if (dim != 2)
    {
        printf("Error: Input is not a matrix but a rank-%i tensor!\n", dim);
        return NAN;
    }
    
    // check if the input is a square matrix
    if (mat->shape[0] != mat->shape[1])
    {
        printf("Error: Input isn't a square matrix but a %ix%i one!\n", mat->shape[0], mat->shape[1]);
        return NAN;
    }

    tensor *cloneMat = copyTensor(mat);
    if (cloneMat == NULL)
    {
        printf("Error: Failed to clone input tensor!\n");
        return NAN;
    }

    // this is for handling the effect of swapping rows when performing Gaussian elimination
    int swapSign = 1;
    
    // gaussian eliminationt
    gaussianElimination(cloneMat, 0, 0, &swapSign);

    // mutliply pivots to get the determinant
    double determinant = 1;
    int maxRowIdx = cloneMat->shape[0];
    int maxColIdx = cloneMat->shape[1];
    for (int rowIdx = 0; rowIdx < maxRowIdx; ++rowIdx)
    {
        // if the current matrix element is 0, return 0
        double elementValue = *(cloneMat->data + rowIdx * (maxColIdx + 1));
        if (elementValue == 0)
        {
            destroyTensor(cloneMat);
            return 0;
        }

        determinant *= elementValue;
    }
    determinant *= swapSign;

    destroyTensor(cloneMat);
    return  determinant;
}

tensor *tensorInverse(const tensor *ten)
{
    // check if the input tensor is valid
    // != NULL
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return NULL;
    }
    
    // tensor of rank > 2
    if (ten->dimensions < 2)
    {
        printf("Error: Cannot compute inversion of a rank-%i tensor!\n", ten->dimensions);
        return NULL;
    }

    // square matrix
    int dims = ten->dimensions;
    int matRows = ten->shape[dims - 2];
    int matCols = ten->shape[dims - 1];
    if (matRows != matCols)
    {
        printf("Error: Cannot inverse matrix of size %ix%i!\n", matRows, matCols);
        return NULL;
    }
    int matSize = matRows;

    // calculate batch dimensions
    int totBatch = 1;
    for (int b = 0; b < dims - 2; ++b)
    {
        totBatch *= ten->shape[b];
    }

    // create augmented matrix
    int augShape[2] = {matRows, 2 * matCols};
    tensor *augMat = createTensor(2, augShape);
    if (augMat == NULL)
    {
        printf("Error: Failed to create augmented matrix!\n");
        return NULL;
    }
    int augCols = augMat->shape[1];

    // tensor to hold the result
    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        destroyTensor(augMat);
        return NULL;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        destroyTensor(augMat);
        return NULL;
    }
    free(resultShape);

    for (int b  = 0; b < totBatch; ++b)
    {
        int offset = b * matSize * matSize;
        
        // fill datas for the augmented matrix
        for (int rowIdx = 0; rowIdx < matSize; ++rowIdx)
        {
            // original matrix data
            for (int colIdx = 0; colIdx < matSize; ++colIdx)
            {
                *(augMat->data + rowIdx * augCols + colIdx) = *(ten->data + offset + rowIdx * matSize + colIdx);
            }

            // augmented identity matrix
            for (int colIdx = matSize; colIdx < augMat->shape[1]; ++colIdx)
            {
                *(augMat->data + rowIdx * augCols + colIdx) = (rowIdx + matCols == colIdx) ? 1 : 0;
            }
        }

        // Gauss-Jordan elimination
        if (!gaussJordanCore(augMat, matSize, augCols, 0))
        {
            // skip this batch
            for (int i = 0; i <matSize * matSize; ++i)
            {
                *(result->data + offset + i) = 0;
            }
            continue;
        }
        // copy data to the result tensor
        for (int rowIdx = 0; rowIdx < matRows; ++rowIdx)
        {
            for (int colIdx = 0; colIdx < matCols; ++colIdx)
            {
                *(result->data + offset + rowIdx * matSize + colIdx) = *(augMat->data + rowIdx * augCols + matSize + colIdx);
            }
        }
    }

    destroyTensor(augMat);
    return result;
}

void gaussianElimination(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign)
{
    // dummy address on stack
    int dummy = 1;

    // check if the user wants to store the sign of the elimination
    swapSign = (swapSign == NULL)? &dummy : swapSign;

    // call the actual process
    gaussianEliminationCore(mat, entryRowIdx, entryColIdx, swapSign);
}

bool tensorTransposeInPlace(tensor *ten, const int *axes)
{
    int *resultShape = malloc(ten->dimensions * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return false;
    }

    // reshape the tensor
    for (int i = 0; i < ten->dimensions; ++i)
    {
        resultShape[i] = ten->shape[axes[i]];
    }

    tensor *result = createTensor(ten->dimensions, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to malloc for result tensor!\n");
        free(resultShape);
        return false;
    }
    free(resultShape);

    int *newCoords = malloc(ten->dimensions * sizeof(int));
    if (newCoords == NULL)
    {
        printf("Error: Failed to malloc for newCoords!\n");
        destroyTensor(result);
        return false;
    }

    int *oldCoords = malloc(ten->dimensions * sizeof(int));
    if (oldCoords == NULL)
    {
        printf("Error: Failed to malloc for oldCoords!\n");
        free(newCoords);
        destroyTensor(result);
        return false;
    }

    for (int i = 0; i < result->size; ++i)
    {
        // locate addresses in the new tensor
        // row-major
        int temp = i;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            newCoords[d] = temp % result->shape[d]; // jump this number of steps in dimension d
            temp /= result->shape[d]; // accumulate jumps at dimension d and bring it onto dimension d - 1
        }

        // maps back to the address in the old tensor
        for (int d = 0; d < result->dimensions; ++d)
        {
            oldCoords[axes[d]] = newCoords[d];
        }

        int oldI = 0;
        int multiplier = 1;
        for (int d = result->dimensions - 1; d >= 0; --d)
        {
            oldI += oldCoords[d] * multiplier; // accumulate jumps
            multiplier *= ten->shape[d]; // magnitude of a jump at the next dimension, so-called "weight" of each dimension
        }

        // finally assign to the new tensor's data
        result->data[i] = ten->data[oldI];
    }

    free(newCoords);
    free(oldCoords);

    // assign the input tensor to the result tensor and free the result tensor
    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}

bool tensorInverseInPlace(tensor *ten)
{
        // check if the input tensor is valid
    // != NULL
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return NULL;
    }
    
    // tensor of rank > 2
    if (ten->dimensions < 2)
    {
        printf("Error: Cannot compute inversion of a rank-%i tensor!\n", ten->dimensions);
        return NULL;
    }

    // square matrix
    int dims = ten->dimensions;
    int matRows = ten->shape[dims - 2];
    int matCols = ten->shape[dims - 1];
    if (matRows != matCols)
    {
        printf("Error: Cannot inverse matrix of size %ix%i!\n", matRows, matCols);
        return NULL;
    }
    int matSize = matRows;

    // calculate batch dimensions
    int totBatch = 1;
    for (int b = 0; b < dims - 2; ++b)
    {
        totBatch *= ten->shape[b];
    }

    // create augmented matrix
    int augShape[2] = {matRows, 2 * matCols};
    tensor *augMat = createTensor(2, augShape);
    if (augMat == NULL)
    {
        printf("Error: Failed to create augmented matrix!\n");
        return NULL;
    }
    int augCols = augMat->shape[1];

    // tensor to hold the result
    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        destroyTensor(augMat);
        return NULL;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        destroyTensor(augMat);
        return NULL;
    }
    free(resultShape);

    for (int b  = 0; b < totBatch; ++b)
    {
        int offset = b * matSize * matSize;
        
        // fill datas for the augmented matrix
        for (int rowIdx = 0; rowIdx < matSize; ++rowIdx)
        {
            // original matrix data
            for (int colIdx = 0; colIdx < matSize; ++colIdx)
            {
                *(augMat->data + rowIdx * augCols + colIdx) = *(ten->data + offset + rowIdx * matSize + colIdx);
            }

            // augmented identity matrix
            for (int colIdx = matSize; colIdx < augMat->shape[1]; ++colIdx)
            {
                *(augMat->data + rowIdx * augCols + colIdx) = (rowIdx + matCols == colIdx) ? 1 : 0;
            }
        }

        // Gauss-Jordan elimination
        if (!gaussJordanCore(augMat, matSize, augCols, 0))
        {
            // skip this batch
            for (int i = 0; i <matSize * matSize; ++i)
            {
                *(result->data + offset + i) = 0;
            }
            continue;
        }
        // copy data to the result tensor
        for (int rowIdx = 0; rowIdx < matRows; ++rowIdx)
        {
            for (int colIdx = 0; colIdx < matCols; ++colIdx)
            {
                *(result->data + offset + rowIdx * matSize + colIdx) = *(augMat->data + rowIdx * augCols + matSize + colIdx);
            }
        }
    }

    destroyTensor(augMat);
    
    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}

// ====================================================================================================
// HELPERS
void matrixMultiplyCore(tensor *result, const double *data1, const double *data2, int M, int K, int N, int batch, bool trans1, bool trans2)
{
    // offset
    int offsetFirst = batch * M * K; // batch * size of the first matrix
    int offsetSecond = batch * K * N; // batch * size of the second matrix
    int offsetResult = batch * M * N; // batch * size of the result matrix

    if (!trans1 && !trans2)
    {
        for (int i = 0; i < M; ++i)
        {
            int firstBase = offsetFirst + i * K;
            int resultBase = offsetResult + i * N;
            for (int k = 0; k < K; ++k)
            {
                int indexFirst = firstBase + k;
                int secondBase = offsetSecond + k * N;
                for (int j = 0; j < N; ++j)
                {
                    int indexSecond = secondBase + j;
                    int indexResult = resultBase + j;
                    result->data[indexResult] += data1[indexFirst] * data2[indexSecond];
                }
            }
        }
    }
    else if (trans1 && !trans2)
    {
        for (int i = 0; i < M; ++i)
        {
            int firstBase = offsetFirst + i;
            int resultBase = offsetResult + i * N;
            for (int k = 0; k < K; ++k)
            {
                int secondBase = offsetSecond + k * N;
                int indexFirst = firstBase + k * M;
                for (int j = 0; j < N; ++j)
                {
                    int indexSecond = secondBase + j;
                    int indexResult = resultBase + j;
                    result->data[indexResult] += data1[indexFirst] * data2[indexSecond];
                }
            }
        }
    }
    else if (!trans1 && trans2)
    {
        for (int i = 0; i < M; ++i)
        {
            int firstBase = offsetFirst + i * K;
            int resultBase = offsetResult + i * N;
            for (int k = 0; k < K; ++k)
            {
                int secondBase = offsetSecond + k;
                int indexFirst = firstBase + k;
                for (int j = 0; j < N; ++j)
                {
                    int indexSecond = secondBase + j * K;
                    int indexResult = resultBase + j;
                    result->data[indexResult] += data1[indexFirst] * data2[indexSecond];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < M; ++i)
        {
            int firstBase = offsetFirst + i;
            int resultBase = offsetResult + i * N;
            for (int k = 0; k < K; ++k)
            {
                int secondBase = offsetSecond + k;
                int indexFirst = firstBase + k * M;
                for (int j = 0; j < N; ++j)
                {
                    int indexSecond = secondBase + j * K;
                    int indexResult = resultBase + j;
                    result->data[indexResult] += data1[indexFirst] * data2[indexSecond];
                }
            }
        }        
    }
}

// Gaussian elimination for MxN matrix
static void gaussianEliminationCore(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign)
{
    // static const int DOUBLE_SIZE = sizeof(double);
    // 0. base case: entryRow == mat->shape[0] - 1 || entryCol == mat->shape[1] - 1
    int maxRowIdx = mat->shape[0];
    int maxColIdx = mat->shape[1];
    if (entryRowIdx >= maxRowIdx || entryColIdx >= maxColIdx)
    {
        return;
    }
    int matCols = maxColIdx;

    // 1. search down in the entry col, find the row i which will give
    //    maximum entry.
    int rowIdxWithMaxVal = entryRowIdx;
    double maxAbsValue = fabs(*(mat->data + entryRowIdx * matCols + entryColIdx));
    for (int rowIdx = entryRowIdx + 1; rowIdx < maxRowIdx; ++rowIdx)
    {
        double absVal = fabs(*(mat->data + rowIdx * matCols + entryColIdx));
        if (absVal > maxAbsValue)
        {
            maxAbsValue = absVal;
            rowIdxWithMaxVal = rowIdx;
        }
    }

    // 2. swap that mat[entryRow] with mat[i] (if i != entryRow)
    if (rowIdxWithMaxVal != entryRowIdx)
    {
        *swapSign = -*swapSign;
        for (int colIdx = entryColIdx; colIdx < maxColIdx; ++colIdx)
        {
            double tempVal = *(mat->data + entryRowIdx * matCols + colIdx);
            *(mat->data + entryRowIdx * matCols + colIdx) = *(mat->data + rowIdxWithMaxVal * matCols + colIdx);
            *(mat->data + rowIdxWithMaxVal * matCols + colIdx) = tempVal;
        }
    }

    // check if the entire col is nonzero
    double pivotVal = *(mat->data + entryRowIdx * matCols + entryColIdx);
    if (pivotVal != 0)
    {
        // 3. loop from mat[entryRow + 1] through and to mat[mat->space[0] - 1], at each iteration i:
        for (int rowIdx = entryRowIdx + 1; rowIdx < maxRowIdx; ++rowIdx)
        {
            //    3.1. find the value u such that:
            //         mat[i][entryCol] == mat[entryRow][entryCol] * u
            double u = *(mat->data + rowIdx * matCols + entryColIdx) / *(mat->data + entryRowIdx * matCols + entryColIdx);

            //    3.2. loop from mat[i][entryCol] through and to mat[i][mat->shape[1] - 1],
            //         at each iteration j: assign mat[i][j] -= mat[entryRow][j] * u
            for (int colIdx = entryColIdx; colIdx < maxColIdx; ++colIdx)
            {
                *(mat->data + rowIdx * matCols + colIdx) -= *(mat->data + entryRowIdx * matCols + colIdx) * u;
            }
        }
    }

    // 4. move to the next pivot and recursive
    gaussianEliminationCore(mat, entryRowIdx + 1, entryColIdx + 1, swapSign);
    return;
}

static bool gaussJordanCore(tensor *augMat, int matSize, int augCols, int entryIdx)
{
    // recursive core:
    // 0. if reached the final diagonal element, return
    if (entryIdx >= matSize)
    {
        return true;
    }

    // 1. loop down from the current row, find one whose absValue is largest, then swap rows
    int rowIdxWithLargestAbsVal = entryIdx;
    double largestAbsVal = fabs(*(augMat->data + entryIdx * (augCols + 1)));
    for (int i = entryIdx; i < matSize; ++i)
    {
        double absVal = fabs(*(augMat->data + i * augCols + entryIdx));
        if (absVal > largestAbsVal)
        {
            largestAbsVal = absVal;
            rowIdxWithLargestAbsVal = i;
        }
    }

    // check if the matrix is a singular matrix
    if (largestAbsVal == 0)
    {
        printf("The matrix is a singular matrix!\n");
        return false;
    }

    if (rowIdxWithLargestAbsVal != entryIdx)
    {
        for (int colIdx = entryIdx; colIdx < augCols; ++colIdx)
        {
            double temp = *(augMat->data + entryIdx * augCols + colIdx);
            *(augMat->data + entryIdx * augCols + colIdx) = *(augMat->data + rowIdxWithLargestAbsVal * augCols + colIdx);
            *(augMat->data + rowIdxWithLargestAbsVal * augCols + colIdx) = temp;
        }
    }

    // 2. turn the current entry to 1
    double u = 1.0 / *(augMat->data + entryIdx * (augCols + 1));
    for (int colIdx = entryIdx; colIdx < augCols; ++colIdx)
    {
        *(augMat->data + entryIdx * augCols + colIdx) *= u;
    }

    // 3. turn all above and below rows of the same col to 0
    for (int rowIdx = 0; rowIdx < matSize; ++rowIdx)
    {
        if (rowIdx == entryIdx)
        {
            continue;
        }
        else if (*(augMat->data + rowIdx * augCols + entryIdx) == 0)
        {
            continue;
        }

        double a = *(augMat->data + rowIdx * augCols + entryIdx);
        for (int colIdx = entryIdx; colIdx < augCols; ++colIdx)
        {
            *(augMat->data + rowIdx * augCols + colIdx) -= a * *(augMat->data + entryIdx * augCols + colIdx);
        }
    }

    // 4. recursive call to the next entry
    return gaussJordanCore(augMat, matSize, augCols, entryIdx + 1);
}