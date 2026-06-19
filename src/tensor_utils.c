#include "../include/tenCor.h"
#include <stdbool.h>
#include <string.h>

tensor *tensorReshape(const tensor *ten, int *newShape, int newDim)
{
    if (ten == NULL)
    {
        printf("Error: Cannot reshape NULL!\n");
        return NULL;
    }

    tensor *result = createTensor(newDim, newShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }

    if (result->size != ten->size)
    {
        printf("Error: Dissimilar size!\nInput tensor has size %i while new tensor has size %i\n", ten->size, result->size);
        destroyTensor(result);
        return NULL;
    }

    // allocate data
    memcpy(result->data, ten->data, ten->size * sizeof(double));

    return result;
}

tensor *tensorFlatten(const tensor *ten)
{
    if (ten == NULL)
    {
        printf("Error: Cannot flatten NULL!\n");
        return NULL;
    }

    int resultShape[1] = {ten->size};

    tensor *result = createTensor(1, resultShape);
    if (result == NULL)
    {
        printf("Error: Failed to create result tensor!\n");
        return NULL;
    }
    memcpy(result->data, ten->data, ten->size * sizeof(double));
    return result;
}

bool tensorReshapeInPlace(tensor *ten, int *newShape, int newDim)
{
    if (ten == NULL)
    {
        printf("Error: Cannot reshape NULL!\n");
        return false;
    }

    // check validity of input
    int newSize = 1;
    for (int d = 0; d < newDim; ++d)
    {
        newSize *= newShape[d];
    }
    if (newSize != ten->size)
    {
        printf("Error: Dissimilar size!\n");
        return false;
    }

    free(ten->shape);
    int *newShapeCopy = malloc(newDim * sizeof(int));
    if (newShapeCopy == NULL)
    {
        printf("Error: Failed to malloc for newShapeCopy!\n");
        return false;
    }
    memcpy(newShapeCopy, newShape, newDim * sizeof(int));
    ten->dimensions = newDim;
    ten->shape = newShapeCopy;

    return true;
}

bool tensorFlattenInPlace(tensor *ten)
{
    if (ten == NULL)
    {
        printf("Error: Cannot flatten NULL!\n");
        return false;
    }

    int *resultShape = malloc(sizeof(int));
    if (resultShape == NULL)
    {
        printf("Error: Failed to malloc for resultShape!\n");
        return false;
    }
    resultShape[0] = ten->size;
    free(ten->shape);
    ten->dimensions = 1;
    ten->shape = resultShape;
    return true;
}
