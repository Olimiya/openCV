#ifndef IMAGEPROCESSALGORITHM_H
#define IMAGEPROCESSALGORITHM_H

#include <QImage>
#include <QMutex>
#define NOISE 0.2
#define FOURIER_FACTOR 14.0;
struct ThreadParam
{
    QImage *src;
    int startIndex;
    int endIndex;
    int maxSpan;//为模板中心到边缘的距离

    void *ctx;//储存特定参数，如高斯噪声参数；

    ThreadParam(void):
        src(nullptr),
        startIndex(0),endIndex(0),
        maxSpan(1),
        ctx(nullptr){}
};

class ImageProcessAlgorithm
{
public:
    /**
     * @brief medianFilter
     * 自适应中值滤波
     * @details
     * 由小模板开始，进行中值排序，若中值不符合判定结果（全为最小或最大）则认为模板过小，增加模板
     * 找到判定成功的模板时，若原来的值不是最小或最大，则保持原值，否则设置为排序后的中间值
     * @param param
     * @return
     */
    static uint medianFilter(ThreadParam *param);
    /**
     * @brief medianFilterCUDA
     * 使用CUDA并行计算的中值滤波器
     * 只需要调用一次该函数
     * @param param
     * @return
     */
    static uint medianFilterCUDA(ThreadParam *param);
    /**
     * @brief saltAndPepperNoise
     * 椒盐噪声
     * @details
     * 对每个像素任意取极大或极小（设定一个阈值内，如0.2）
     * @param param
     * @return
     */
    static uint saltAndPepperNoise(ThreadParam *param);
    /**
     * @brief gaussianNoise
     * 高斯噪声
     * @details
     * 对每个像素值增加一个符合高斯分布的值，其中一系列高斯分布的值有Box-Muller产生，需指定均值、标准差
     * @param param
     * @return
     */
    static uint gaussianNoise(ThreadParam *param);

    /**
     * @brief ArithMeanFilter
     * 算术均值滤波器（线性平滑滤波器）3*3模板
     * @details
     * 在kernel中计算算术均值，作为像素值
     * @param param
     * @return
     */
    static uint ArithMeanFilter(ThreadParam *param);
    static uint ArithMeanFilterCUDA(ThreadParam *param);

    /**
     * @brief gaussianFilter
     * 高斯滤波器 3*3模板
     * @param param
     * @details
     * 首先获取高斯分布的kernel，即通过二维高斯分布计算，
     * 然后利用kernel进行卷积，作为像素值
     * @return
     */
    static uint gaussianFilter(ThreadParam *param);
    /**
     * @brief WienerFilter
     * 维纳滤波器 自适应 3*3邻域
     * @param param
     * @details
     * 首先计算所有像素一定邻域的均值、标准差，拟合得到噪声值，
     * 然后利用均值、标准差和噪声值的结果计算出像素值
     * @return
     */
    static uint WienerFilter(ThreadParam *param);

    /**
     * @brief BicubicRotate
     * 双三次插值旋转
     * 使用埃尔米特双三次插值公式
     * @see
     * https://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
     * @param param
     * @return
     */
    static uint BicubicRotate(ThreadParam *param);
    /**
     * @brief BicubicScale
     * 双三次插值放缩
     * 使用Keys卷积核模板
     * @see
     * https://en.wikipedia.org/wiki/Bicubic_interpolation#/media/File:Comparison_of_1D_and_2D_interpolation.svg
     * @param param
     * @return
     */
    static uint BicubicScale(ThreadParam *param);

    /**
     * @brief FourierTransform
     * 二维离散傅里叶变换
     * 暴力法
     * @param param
     * @return
     */
    static uint FourierTransform(ThreadParam *param);

    struct GassuianParam
    {
        double mean;
        double stddev;
    };
    struct ScaleParams
    {
        QImage *src;
        double scale;
    };

    struct RotateParams
    {
        QImage *src;
        double angle;//rad
    };

    template <typename T>
    struct vec2
    {
        T x,y;
    };
private:
    //自适应中值滤波判定中间值
    static bool GetValue(int p[], int size, int &value);
    //Box-Muller变换求高斯分布值
    static double BoxMullerGenerator(double mean, double stddev);
    //计算高斯模板
    static void GetGaussianTemplate(double t[3][3], double stddev);
    //双三次插值埃尔米特公式
    static double BicubicHermite(double A, double B, double C, double D, double t);
    //双三次插值kernel模板计算
    static inline double BicubicWeight(double x);

    static std::vector<int> getPixelHelp(QImage *image, int x, int y);
};

#endif // IMAGEPROCESSALGORITHM_H
