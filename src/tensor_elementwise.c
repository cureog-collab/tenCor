#include "../include/tenCor.h"
#include <stdlib.h>

void tensorScale(tensor *ten, double scalar)
{
    if (isnan(scalar) || isinf(scalar))
    {
        return;
    }

    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] *= scalar;
    }
}

tensor *tensorAdd(const tensor *ten1, const tensor *ten2, bool isAdd)
{
    int dims1 = ten1->dimensions;
    int dims2 = ten2->dimensions;
    const tensor *ptrToBigTensor = (dims1 > dims2) ? ten1 : ten2;
    const tensor *ptrToSmallTensor = (dims2 < dims1) ? ten2 : ten1;
    int maxDims = ptrToBigTensor->dimensions;
    int dimDiff = maxDims - ptrToSmallTensor->dimensions;
    int shapeSmallPadded[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        shapeSmallPadded[d] = (d < dimDiff) ? 1 : ptrToSmallTensor->shape[d - dimDiff];
    }

    int resultShape[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        int dimBig = ptrToBigTensor->shape[d];
        int dimSmall = shapeSmallPadded[d];
        if (dimBig == dimSmall)
        {
            resultShape[d] = dimBig;
        }
        else if (dimBig == 1)
        {
            resultShape[d] = dimSmall;
        }
        else if (dimSmall == 1)
        {
            resultShape[d] = dimBig;
        }
        else
        {
            printf("Error: Tensors are not broadcastable!\n");
            return NULL;
        }
    }

    tensor *result = createTensor(maxDims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    if (isAdd)
    {
        for (int i = 0; i < result->size; ++i)
        {
            int temp = i;
            int idxBig = 0;
            int idxSmall = 0;
            int jumpBig = 1;
            int jumpSmall = 1;
            for (int d = maxDims -1; d >=0; --d)
            {
                int coord = temp % resultShape[d];
                temp /= resultShape[d];

                int coordBig = coord % ptrToBigTensor->shape[d];
                int coordSmall = coord % shapeSmallPadded[d];

                idxBig += coordBig * jumpBig;
                idxSmall += coordSmall * jumpSmall;

                jumpBig *= ptrToBigTensor->shape[d];
                jumpSmall *= shapeSmallPadded[d];
            }
            result->data[i] = ptrToBigTensor->data[idxBig] + ptrToSmallTensor->data[idxSmall];
        }
    }
    else
    {
        int minusSign = (dims1 > dims2) ? 1 : -1;
        for (int i = 0; i < result->size; ++i)
        {
            int temp = i;
            int idxBig = 0;
            int idxSmall = 0;
            int jumpBig = 1;
            int jumpSmall = 1;
            for (int d = maxDims -1; d >=0; --d)
            {
                int coord = temp % resultShape[d];
                temp /= resultShape[d];

                int coordBig = coord % ptrToBigTensor->shape[d];
                int coordSmall = coord % shapeSmallPadded[d];

                idxBig += coordBig * jumpBig;
                idxSmall += coordSmall * jumpSmall;

                jumpBig *= ptrToBigTensor->shape[d];
                jumpSmall *= shapeSmallPadded[d];
            }
            result->data[i] = minusSign * (ptrToBigTensor->data[idxBig] - ptrToSmallTensor->data[idxSmall]);
        }   
    }

    return result;
}

tensor *tensorAddBias(const tensor *ten, const tensor *bias)
{
    if (ten == NULL || bias == NULL)
    {
        printf("Error: Cannot perform tensorAddBias with NULL!\n");
        return NULL;
    }
    else if (bias->dimensions != 1)
    {
        printf("Error: Bias must be a vector (currently it is a rank-%i tensor).", bias->dimensions);
        return NULL;
    }
    else if (bias->shape[0] != ten->shape[ten->dimensions - 1])
    {
        printf("Error: Last dimensions of input tensor and bias vector must have the same number of elements!\n");
        return NULL;
    }

    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    int tenSize = ten->size;
    int biasSize = bias->size;

    // divide the 1D RAM data into contiguous blocks of biasSize
    int numBlocks = tenSize / biasSize;

    // jump into each block
    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx)
    {
        // address of the block
        int blockOffset = blockIdx * biasSize;

        // loop inside that block
        for (int i = 0; i < biasSize; ++i)
        {
            *(result->data + blockOffset + i) = *(ten->data + blockOffset + i) + *(bias->data + i);
        }
    }

    return result;
}

tensor *tensorDivide(const tensor *ten1, const tensor *ten2)
{
    int dims1 = ten1->dimensions;
    int dims2 = ten2->dimensions;
    const tensor *ptrToBigTensor = (dims1 > dims2) ? ten1 : ten2;
    const tensor *ptrToSmallTensor = (dims2 < dims1) ? ten2 : ten1;
    int maxDims = ptrToBigTensor->dimensions;
    int dimDiff = maxDims - ptrToSmallTensor->dimensions;
    int shapeSmallPadded[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        shapeSmallPadded[d] = (d < dimDiff) ? 1 : ptrToSmallTensor->shape[d - dimDiff];
    }

    int resultShape[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        int dimBig = ptrToBigTensor->shape[d];
        int dimSmall = shapeSmallPadded[d];
        if (dimBig == dimSmall)
        {
            resultShape[d] = dimBig;
        }
        else if (dimBig == 1)
        {
            resultShape[d] = dimSmall;
        }
        else if (dimSmall == 1)
        {
            resultShape[d] = dimBig;
        }
        else
        {
            printf("Error: Tensors are not broadcastable!\n");
            return NULL;
        }
    }

    tensor *result = createTensor(maxDims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    bool isTen1Bigger = dims1 > dims2;
    if (isTen1Bigger)
    {
        for (int i = 0; i < result->size; ++i)
        {
            int temp = i;
            int idxBig = 0;
            int idxSmall = 0;
            int jumpBig = 1;
            int jumpSmall = 1;
            for (int d = maxDims -1; d >=0; --d)
            {
                int coord = temp % resultShape[d];
                temp /= resultShape[d];

                int coordBig = coord % ptrToBigTensor->shape[d];
                int coordSmall = coord % shapeSmallPadded[d];

                idxBig += coordBig * jumpBig;
                idxSmall += coordSmall * jumpSmall;

                jumpBig *= ptrToBigTensor->shape[d];
                jumpSmall *= shapeSmallPadded[d];
            }

            double valBig = ptrToBigTensor->data[idxBig];
            double valSmall = ptrToSmallTensor->data[idxSmall];
            result->data[i] = valBig / (valSmall + EPSILON);
        }
    }
    else  
    {
        for (int i = 0; i < result->size; ++i)
        {
            int temp = i;
            int idxBig = 0;
            int idxSmall = 0;
            int jumpBig = 1;
            int jumpSmall = 1;
            for (int d = maxDims -1; d >=0; --d)
            {
                int coord = temp % resultShape[d];
                temp /= resultShape[d];

                int coordBig = coord % ptrToBigTensor->shape[d];
                int coordSmall = coord % shapeSmallPadded[d];

                idxBig += coordBig * jumpBig;
                idxSmall += coordSmall * jumpSmall;

                jumpBig *= ptrToBigTensor->shape[d];
                jumpSmall *= shapeSmallPadded[d];
            }

            double valBig = ptrToBigTensor->data[idxBig];
            double valSmall = ptrToSmallTensor->data[idxSmall];
            result->data[i] = valSmall / (valBig + EPSILON);
        }
    }
    return result;
}

tensor *tensorHadamardProduct(const tensor *ten1, const tensor *ten2)
{
    int dims1 = ten1->dimensions;
    int dims2 = ten2->dimensions;
    const tensor *ptrToBigTensor = (dims1 > dims2) ? ten1 : ten2;
    const tensor *ptrToSmallTensor = (dims2 < dims1) ? ten2 : ten1;
    int maxDims = ptrToBigTensor->dimensions;
    int dimDiff = maxDims - ptrToSmallTensor->dimensions;
    int shapeSmallPadded[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        shapeSmallPadded[d] = (d < dimDiff) ? 1 : ptrToSmallTensor->shape[d - dimDiff];
    }

    int resultShape[maxDims];
    for (int d = 0; d < maxDims; ++d)
    {
        int dimBig = ptrToBigTensor->shape[d];
        int dimSmall = shapeSmallPadded[d];
        if (dimBig == dimSmall)
        {
            resultShape[d] = dimBig;
        }
        else if (dimBig == 1)
        {
            resultShape[d] = dimSmall;
        }
        else if (dimSmall == 1)
        {
            resultShape[d] = dimBig;
        }
        else
        {
            printf("Error: Tensors are not broadcastable!\n");
            return NULL;
        }
    }

    tensor *result = createTensor(maxDims, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    for (int i = 0; i < result->size; ++i)
    {
        int temp = i;
        int idxBig = 0;
        int idxSmall = 0;
        int jumpBig = 1;
        int jumpSmall = 1;
        for (int d = maxDims -1; d >=0; --d)
        {
            int coord = temp % resultShape[d];
            temp /= resultShape[d];

            int coordBig = coord % ptrToBigTensor->shape[d];
            int coordSmall = coord % shapeSmallPadded[d];

            idxBig += coordBig * jumpBig;
            idxSmall += coordSmall * jumpSmall;

            jumpBig *= ptrToBigTensor->shape[d];
            jumpSmall *= shapeSmallPadded[d];
        }
        result->data[i] = ptrToBigTensor->data[idxBig] * ptrToSmallTensor->data[idxSmall];
    }
    return result;
}

tensor *tensorRelu(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = (ten->data[i] > 0) ? ten->data[i] : 0;
    }

    return result;
}

tensor *tensorSigmoid(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = 1.0 / (1 + exp(-ten->data[i]));
    }

    return result;
}

tensor *tensorTanh(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = tanh(ten->data[i]);
    }

    return result;
}

tensor *tensorReluDerivative(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    for (int i = 0; i < ten->size; ++i)
    {
        result->data[i] = (ten->data[i] > 0) ? 1.0 : 0;
    }

    return result;
}

tensor *tensorSigmoidDerivative(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    for (int i = 0; i < ten->size; ++i)
    {
        double value = ten->data[i];
        result->data[i] = value * (1.0 - value);
    }

    return result;
}

tensor *tensorTanhDerivative(const tensor *ten)
{
    tensor *result = createTensor(ten->dimensions, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    for (int i = 0; i < ten->size; ++i)
    {
        double value = ten->data[i];
        result->data[i] = 1.0 - value * value;
    }

    return result;
}

tensor *tensorSoftmax(const tensor *ten, int axis)
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

    tensor *result = createTensor(dims, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

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
            int baseIdx = i * axisSize * innerSize + j;

            // first loop inside axis: find maxVal
            double maxVal = ten->data[baseIdx];
            for (int k = 1; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;
                maxVal = (ten->data[currIdx] > maxVal) ? ten->data[currIdx] : maxVal;
            }

            // second loop inside axis: compute exp(value - maxVal) & compute sum of all exp(value - maxVal)
            double partitionFuncion = 0.0;
            for (int k = 0; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;

                // temporarily store into result->data
                double temp = exp(ten->data[currIdx] - maxVal);
                result->data[currIdx] = temp;
                partitionFuncion += temp;
            }

            // third loop inside axis: c
            double invZ = 1.0 / partitionFuncion;
            for (int k = 0; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;
                result->data[currIdx] *= invZ;
            }
        }
    }

    return result;
}

bool tensorAddBiasInPlace(tensor *ten, const tensor *bias)
{
    if (ten == NULL || bias == NULL)
    {
        printf("Error: Cannot perform tensorAddBias with NULL!\n");
        return false;
    }
    else if (bias->dimensions != 1)
    {
        printf("Error: Bias must be a vector (currently it is a rank-%i tensor).", bias->dimensions);
        return false;
    }
    else if (bias->shape[0] != ten->shape[ten->dimensions - 1])
    {
        printf("Error: Last dimensions of input tensor and bias vector must have the same number of elements!\n");
        return false;
    }

    int tenSize = ten->size;
    int biasSize = bias->size;

    // divide the 1D RAM data into contiguous blocks of biasSize
    int numBlocks = tenSize / biasSize;

    // jump into each block
    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx)
    {
        // address of the block
        int blockOffset = blockIdx * biasSize;

        // loop inside that block
        for (int i = 0; i < biasSize; ++i)
        {
            *(ten->data + blockOffset + i) += *(bias->data + i);
        }
    }

    return true;
}

void tensorReluInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = (ten->data[i] > 0) ? ten->data[i] : 0;
    }
}

void tensorSigmoidInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = 1.0 / (1 + exp(-ten->data[i]));
    }
}

void tensorTanhInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = tanh(ten->data[i]);
    }
}

bool tensorSoftmaxInPlace(tensor *ten, int axis)
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

    tensor *result = createTensor(dims, ten->shape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return false;
    }

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
            int baseIdx = i * axisSize * innerSize + j;

            // first loop inside axis: find maxVal
            double maxVal = ten->data[baseIdx];
            for (int k = 1; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;
                maxVal = (ten->data[currIdx] > maxVal) ? ten->data[currIdx] : maxVal;
            }

            // second loop inside axis: compute exp(value - maxVal) & compute sum of all exp(value - maxVal)
            double partitionFuncion = 0.0;
            for (int k = 0; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;

                // temporarily store into result->data
                double temp = exp(ten->data[currIdx] - maxVal);
                result->data[currIdx] = temp;
                partitionFuncion += temp;
            }

            // third loop inside axis: c
            double invZ = 1.0 / partitionFuncion;
            for (int k = 0; k < axisSize; ++k)
            {
                int currIdx = baseIdx + k * innerSize;
                result->data[currIdx] *= invZ;
            }
        }
    }

    free(ten->data);
    free(ten->shape);
    *ten = *result;
    free(result);
    return true;
}

void tensorReluDerivativeInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        ten->data[i] = (ten->data[i] > 0) ? 1.0 : 0;
    }
}

void tensorSigmoidDerivativeInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        double value = ten->data[i];
        ten->data[i] = value * (1.0 - value);
    }
}

void tensorTanhDerivativeInPlace(tensor *ten)
{
    for (int i = 0; i < ten->size; ++i)
    {
        double value = ten->data[i];
        ten->data[i] = 1.0 - value * value;
    }
}

// ==========================================================================================
// HELPERS

// check if the two tensors are the same shape
bool checkShapeSim(const tensor *ten1, const tensor *ten2)
{
    if (ten1->dimensions != ten2->dimensions)
    {
        printf("Error: Dimensions of the two tensors are not equal: %i and %i.\n", ten1->dimensions, ten2->dimensions);
        return false;
    }
    if (memcmp(ten1->shape, ten2->shape, ten1->dimensions * sizeof(int)) != 0)
    {
        printf("Error: Shapes of the two tensors are not equal!\n");
        return false;
    }
    return true;
}