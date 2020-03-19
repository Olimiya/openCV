#include "imageprocessalgorithm.h"

uint ImageProcessAlgorithm::medianFilter(ThreadParam *param)
{
    return 0;
}

uint ImageProcessAlgorithm::addNoise(ThreadParam *param)
{
    int maxWidth = param->src->width();
    int maxHeight = param->src->height();

    int startIndex = param->startIndex;
    int endIndex = param->endIndex;
    unsigned char* pRealData = (unsigned char*)param->src->bits();
//    int bitCount = param->src->bitPlaneCount() / 8;
    int bitCount = 32 / 8;
    ///GetBPP() / 8;
    int pit = param->src->bytesPerLine();
    ///GetPitch();

    auto src = param->src;
    srand(time(0));
    for (int i = startIndex; i <= endIndex; ++i)
    {
        int x = i % maxWidth;
        int y = i / maxWidth;
        if ((rand() % 1000) * 0.001 < NOISE)
        {
            int value = 0;
            if (rand() % 1000 < 500)
            {
                value = 0;
            }
            else
            {
                value = 255;
            }
//            param->src->setPixelColor(x, y, QColor(value, value, value));
            if (bitCount == 1)
            {

                *(pRealData + pit * y + x * bitCount) = value;
            }
            else
            {
                *(pRealData + pit * y + x * bitCount) = value;
                *(pRealData + pit * y + x * bitCount + 1) = value;
                *(pRealData + pit * y + x * bitCount + 2) = value;
            }
        }
    }
    return 0;
}
