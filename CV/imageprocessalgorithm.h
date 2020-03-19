#ifndef IMAGEPROCESSALGORITHM_H
#define IMAGEPROCESSALGORITHM_H

#include <QImage>
#define NOISE 0.2
struct ThreadParam
{
    QImage *src;
    int startIndex;
    int endIndex;
    int maxSpan;//为模板中心到边缘的距离
};

class ImageProcessAlgorithm
{
public:
    static uint medianFilter(ThreadParam *param);
    static uint addNoise(ThreadParam *param);
};

#endif // IMAGEPROCESSALGORITHM_H
