#include "../include/tenCor.h"
#include <stdlib.h>

double tensorSum(const tensor *ten)
{
    if (ten == NULL)
    {
        printf("Error: Cannot perform tensorSum on NULL!\n");
        return NAN;
    }

    double sum = 0;
    for (int i = 0; i < ten->size; ++i)
    {
        sum += ten->data[i];
    }

    return sum;
}

double tensorMean(const tensor *ten)
{
    double sum = tensorSum(ten);
    if (isnan(sum))
    {
        return NAN;
    }

    return (sum / ten->size);
}

// the below code look ugly, but I think copy-and-pasting here is more efficient
// than code a helper function and then call it in each function.

tensor *tensorSumByAxis(const tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return NULL;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return NULL;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;
            double sum = 0;

            // loop inside axis
            for (int k = 0; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                sum += ten->data[oldCurrIdx];
            }

            result->data[resultIdx] = sum;
        }
    }

    return result;
}

tensor *tensorMaxByAxis(const tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return NULL;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return NULL;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;
            double maxVal = ten->data[oldBaseIdx];

            // loop inside axis
            for (int k = 1; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                maxVal = (ten->data[oldCurrIdx] > maxVal) ? ten->data[oldCurrIdx] : maxVal;
            }

            result->data[resultIdx] = maxVal;
        }
    }

    return result;
}

tensor *tensorArgmax(const tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return NULL;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return NULL;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return NULL;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return NULL;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;

            double maxVal = ten->data[oldBaseIdx];
            int bestK = 0;

            // loop inside axis
            for (int k = 1; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                if (ten->data[oldCurrIdx] > maxVal)
                {
                    maxVal = ten->data[oldCurrIdx];
                    bestK = k;
                }
            }

            result->data[resultIdx] = (double)bestK;
        }
    }

    return result;
}

bool tensorSumByAxisInPlace(tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return false;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return false;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return false;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return false;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;
            double sum = 0;

            // loop inside axis
            for (int k = 0; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                sum += ten->data[oldCurrIdx];
            }

            result->data[resultIdx] = sum;
        }
    }

    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}

bool tensorMaxByAxisInPlace(tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return false;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return false;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return false;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return false;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;
            double maxVal = ten->data[oldBaseIdx];

            // loop inside axis
            for (int k = 1; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                maxVal = (ten->data[oldCurrIdx] > maxVal) ? ten->data[oldCurrIdx] : maxVal;
            }

            result->data[resultIdx] = maxVal;
        }
    }

    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}

bool tensorArgmaxInPlace(tensor *ten, int axis)
{
    if (ten == NULL)
    {
        printf("Error: Input tensor is NULL!\n");
        return false;
    }

    if (axis < 0 || axis >= ten->dimensions)
    {
        printf("Error: Axis %i is not a valid axis!\n", axis);
        return false;
    }

    int dims = ten->dimensions;

    int *resultShape = malloc(dims * sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return false;
    }
    memcpy(resultShape, ten->shape, dims * sizeof(int));
    resultShape[axis] = 1;

    tensor *result = createTensor(dims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        free(resultShape);
        return false;
    }
    free(resultShape);

    // find sizes to later detect elements on axis
    int axisSize = ten->shape[axis];
    int innerSize = 1;
    for (int d = axis + 1; d < dims; ++d)
    {
        innerSize *= ten->shape[d];
    }
    int outerSize = ten->size / (axisSize * innerSize);

    // jump through outer axes
    for (int i = 0; i < outerSize; ++i)
    {
        // jump though inner axes
        for (int j = 0; j < innerSize; ++j)
        {
            // address on the result tensor
            int resultIdx = i * innerSize + j;
            int oldBaseIdx = i * axisSize * innerSize + j;

            double maxVal = ten->data[oldBaseIdx];
            int bestK = 0;

            // loop inside axis
            for (int k = 1; k < axisSize; ++k)
            {
                int oldCurrIdx = oldBaseIdx + k * innerSize;
                if (ten->data[oldCurrIdx] > maxVal)
                {
                    maxVal = ten->data[oldCurrIdx];
                    bestK = k;
                }
            }

            result->data[resultIdx] = (double)bestK;
        }
    }

    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}