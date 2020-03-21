#include "imageprocessalgorithm.h"
#include <algorithm>

bool ImageProcessAlgorithm::GetValue(int p[], int size, int &value)
{
    //数组中间的值
    int zxy = p[(size - 1) / 2];
    //用于记录原数组的下标
    int *a = new int[size];
    int index = 0;
    for (int i = 0; i<size; ++i)
        a[index++] = i;

    for (int i = 0; i<size - 1; i++)
        for (int j = i + 1; j<size; j++)
            if (p[i]>p[j]) {
                int tempA = a[i];
                a[i] = a[j];
                a[j] = tempA;
                int temp = p[i];
                p[i] = p[j];
                p[j] = temp;

            }
    int zmax = p[size - 1];
    int zmin = p[0];
    int zmed = p[(size - 1) / 2];

    if (zmax>zmed&&zmin<zmed) {
        if (zxy>zmin&&zxy<zmax)
            value = (size - 1) / 2;
        else
            value = a[(size - 1) / 2];
        delete[]a;
        return true;
    }
    else {
        delete[]a;
        return false;
    }

}

double ImageProcessAlgorithm::BoxMullerGenerator(double mean, double stddev)
{
    static const double twopi = 2.0 * acos(-1);
    double u1, u2;
    static double z0, z1;
    static bool generate = false;
    generate = !generate;
    if (!generate)
        return z1 * stddev + mean;
    do
    {
        u1 = (double)rand() / RAND_MAX;
        u2 = (double)rand() / RAND_MAX;
    } while (u1 <= DBL_MIN);
    z0 = sqrt(-2.0 * log(u1)) * cos(twopi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(twopi * u2);
    return z0 * stddev + mean;
}

void ImageProcessAlgorithm::GetGaussianTemplate(double t[3][3], double stddev)
{
    const int center = 1; // [[0, 1, 2], [0, 1, 2], [0, 1, 2]], center is (1, 1)
    double total = 0;
    static const double PI = acos(-1);
    for (int i = 0; i < 3; ++i)
    {
        double xsq = pow(i - center, 2.0);
        for (int j = 0; j < 3; ++j)
        {
            double ysq = pow(j - center, 2.0);
            double f = 1 / (2.0 * PI * stddev * stddev);
            double e = exp(-(xsq + ysq) / (2.0 * stddev * stddev));
            t[i][j] = f * e;
            total += t[i][j];
        }
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            t[i][j] /= total;
}

double ImageProcessAlgorithm::BicubicHermite(double A, double B, double C, double D, double t)
{
    double a = -A / 2.0 + (3.0 * B) / 2.0 - (3.0 * C) / 2.0 + D / 2.0;
    double b = A - (5.0 * B) / 2.0 + 2.0 * C - D / 2.0;
    double c = -A / 2.0 + C / 2.0;
    double d = B;
    return a * t * t * t + b * t * t + c * t + d;
}

double ImageProcessAlgorithm::BicubicWeight(double x)
{
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return (a + 2.0)*x*x*x - (a + 3.0)*x*x + 1.0;
    else if (x < 2.0)
        return a * x*x*x - 5.0*a * x*x + 8.0*a * x - 4.0 * a;
    return 0.0;
}

std::vector<int> ImageProcessAlgorithm::getPixelHelp(QImage *image, int x, int y)
{
    QRgb pixelColor = image->pixel(x, y);
    std::vector<int> pixel = {qRed(pixelColor),  qGreen(pixelColor), qBlue(pixelColor)};
    return pixel;
}

uint ImageProcessAlgorithm::medianFilter(ThreadParam *param)
{
    int maxWidth = param->src->width();
    int maxHeight = param->src->height();
    int startIndex = param->startIndex;
    int endIndex = param->endIndex;
    int maxSpan = param->maxSpan;
    int maxLength = (maxSpan * 2 + 1) * (maxSpan * 2 + 1);

    unsigned char* pRealData = (unsigned char*)param->src->bits();
    //    int bitCount = param->src->bitPlaneCount() / 8;
    int bitCount = 32 / 8;
    int pit = param->src->bytesPerLine();

    int *pixel = new int[maxLength];//存储每个像素点的灰度
    int *pixelR = new int[maxLength];
    int *pixelB = new int[maxLength];
    int *pixelG = new int[maxLength];
    int index = 0;
    for (int i = startIndex; i <= endIndex; ++i)
    {
        int Sxy = 1;
        int med = 0;
        int state = 0;
        int x = i % maxWidth;
        int y = i / maxWidth;
        while (Sxy <= maxSpan)
        {
            index = 0;
            for (int tmpY = y - Sxy; tmpY <= y + Sxy && tmpY <maxHeight; tmpY++)
            {
                if (tmpY < 0) continue;
                for (int tmpX = x - Sxy; tmpX <= x + Sxy && tmpX<maxWidth; tmpX++)
                {
                    if (tmpX < 0) continue;
                    if (bitCount == 1)
                    {
                        pixel[index] = *(pRealData + pit*(tmpY)+(tmpX)*bitCount);
                        pixelR[index] = pixel[index];
                        index++;

                    }
                    else
                    {
                        pixelR[index] = *(pRealData + pit*(tmpY)+(tmpX)*bitCount + 2);
                        pixelG[index] = *(pRealData + pit*(tmpY)+(tmpX)*bitCount + 1);
                        pixelB[index] = *(pRealData + pit*(tmpY)+(tmpX)*bitCount);
                        pixel[index] = int(pixelB[index] * 0.299 + 0.587*pixelG[index] + pixelR[index] * 0.144);
                        index++;

                    }
                }

            }
            if (index <= 0)
                break;
            if ((state = GetValue(pixel, index, med)) == 1)
                break;

            Sxy++;
        };

        if (state)
        {
            if (bitCount == 1)
            {
                *(pRealData + pit*y + x*bitCount) = pixelR[med];

            }
            else
            {
                *(pRealData + pit*y + x*bitCount + 2) = pixelR[med];
                *(pRealData + pit*y + x*bitCount + 1) = pixelG[med];
                *(pRealData + pit*y + x*bitCount) = pixelB[med];

            }
        }

    }

    delete[]pixel;
    delete[]pixelR;
    delete[]pixelG;
    delete[]pixelB;
    return 0;
}

uint ImageProcessAlgorithm::saltAndPepperNoise(ThreadParam *param)
{
    int maxWidth = param->src->width();
    //    int maxHeight = param->src->height();

    int startIndex = param->startIndex;
    int endIndex = param->endIndex;
    unsigned char* pRealData = (unsigned char*)param->src->bits();
    //    int bitCount = param->src->bitPlaneCount() / 8;
    int bitCount = 32 / 8;
    ///GetBPP() / 8;
    int pit = param->src->bytesPerLine();
    ///GetPitch();

    //    auto src = param->src;
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

uint ImageProcessAlgorithm::gaussianNoise(ThreadParam *param)
{
    srand((unsigned int)time(0));
    auto gp = (GassuianParam*)(param->ctx);
    QImage *img = param->src;
    for (int idx = param->startIndex; idx < param->endIndex; ++idx)
    {
        int x = idx % img->width();
        int y = idx / img->width();
        QRgb pixelColor = img->pixel(x, y);
        int pixel[3] = {qRed(pixelColor),  qGreen(pixelColor), qBlue(pixelColor)};
        for (int i = 0; i < 3; ++i)
        {
            double val = pixel[i] + BoxMullerGenerator(gp->mean, gp->stddev);
            if (val > 255.0)
                val = 255.0;
            if (val < 0.0)
                val = 0.0;
            pixel[i] = (int)val;
        }
        img->setPixel(x, y, qRgb(pixel[0], pixel[1], pixel[2]));
    }
    return 0;
}

uint ImageProcessAlgorithm::ArithMeanFilter(ThreadParam *param)
{
    QImage *img = param->src;
    const int TEMPLATE_SIZE = 3 * 3;
    for (int idx = param->startIndex; idx < param->endIndex; ++idx)
    {
        int x = idx % img->width();
        int y = idx / img->width();
        // skip the border
        if (x < 1 || y < 1 || x >= img->width() - 1 || y >= img->height() - 1)
        {
            continue;
        }
        int r, g, b;
        r = g = b = 0;

#define ACCUMULATE(_x, _y) { \
    QRgb _color = img->pixel(_x, _y); \
    int _pixel[3] = {qRed(_color),  qGreen(_color), qBlue(_color)}; \
    r += _pixel[0]; g += _pixel[1]; b += _pixel[2]; }
        ACCUMULATE(x - 1, y - 1); ACCUMULATE(x, y - 1); ACCUMULATE(x + 1, y - 1);
        ACCUMULATE(x - 1, y);     ACCUMULATE(x, y);     ACCUMULATE(x + 1, y);
        ACCUMULATE(x - 1, y + 1); ACCUMULATE(x, y + 1); ACCUMULATE(x + 1, y - 1);
#undef ACCUMULATE
        img->setPixel(x, y, qRgb(r / TEMPLATE_SIZE, g / TEMPLATE_SIZE, b / TEMPLATE_SIZE));
    }
    return 0;
}

uint ImageProcessAlgorithm::gaussianFilter(ThreadParam *param)
{
    QImage *img = param->src;
    auto gp = (GassuianParam*)param->ctx;
    const int SIZE = 3;
    double m[SIZE][SIZE];
    GetGaussianTemplate(m, gp->stddev);
    for (int idx = param->startIndex; idx < param->endIndex; ++idx)
    {
        int x = idx % img->width();
        int y = idx / img->width();
        // skip the border
        if (x < 1 || y < 1 || x >= img->width() - 1 || y >= img->height() - 1)
        {
            continue;
        }
        double r, g, b;
        r = g = b = 0.0;
        //        QRgb pixelColor = img->pixel(x, y);
        //        int pixel[3] = {qRed(pixelColor),  qGreen(pixelColor), qBlue(pixelColor)};
#define ACCUMULATE(_x, _y, _a, _b) { \
    QRgb _color = img->pixel(_x, _y); \
    int _pixel[3] = {qRed(_color),  qGreen(_color), qBlue(_color)}; \
    r += (double)_pixel[0]*m[_a][_b]; \
    g += (double)_pixel[1]*m[_a][_b]; \
    b += (double)_pixel[2]*m[_a][_b]; }
        ACCUMULATE(x - 1, y - 1, 0, 0); ACCUMULATE(x, y - 1, 1, 0); ACCUMULATE(x + 1, y - 1, 2, 0);
        ACCUMULATE(x - 1, y   ,  0, 1); ACCUMULATE(x, y    , 1, 1); ACCUMULATE(x + 1, y    , 2, 1);
        ACCUMULATE(x - 1, y + 1, 0, 2); ACCUMULATE(x, y + 1, 1, 2); ACCUMULATE(x + 1, y - 1, 2, 2);
#undef ACCUMULATE
#define CLAMP(v) {if(v>255.0)v=255.0;else if(v<0.0)v=0.0;}
        CLAMP(r); CLAMP(g); CLAMP(b);
#undef CLAMP
        img->setPixel(x, y, qRgb(r, g, b));
    }
    return 0;
}

uint ImageProcessAlgorithm::WienerFilter(ThreadParam *param)
{
#define OFFSET(x, y) (y * img->width() + x - startIndex)
    QImage *img = param->src;
    int startIndex = param->startIndex;
    int endIndex = param->endIndex;
    int len = endIndex - startIndex;
    double noise[3] = {0, 0, 0};
    double *mean[3], *variance[3];
    for (int ch = 0; ch < 3; ++ch)
    {
        mean[ch] = new double[len];
        variance[ch] = new double[len];
    }
    // loop #1: calc mean, var, and noise
    for (int idx = startIndex; idx <endIndex; ++idx)
    {
        int x = idx % img->width();
        int y = idx / img->width();
        auto offset = OFFSET(x, y);
        // skip the border
        if (x < 1 || y < 1 || x >= img->width() - 1 || y >= img->height() - 1)
        {
            continue;
        }
        int *pixels[9];
        for (int i = -1, c = 0; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j, ++c)
            {
                auto color = img->pixel(x + i, y + j);
                pixels[c] = new int[3]{qRed(color), qGreen(color), qBlue(color)};
            }
        for (int ch = 0; ch < 3; ++ch) // RGB channels
        {
            mean[ch][offset] = 0.0;
            variance[ch][offset] = 0.0;
            for (int i = 0; i < 9; ++i)
                mean[ch][offset] += pixels[i][ch];
            mean[ch][offset] /= 9;
            for (int i = 0; i < 9; ++i)
                variance[ch][offset] += pow(pixels[i][ch] - mean[ch][offset], 2.0);
            variance[ch][offset] /= 9;
            noise[ch] += variance[ch][offset];
        }
        for(int i = 0; i < 9; i++)
            delete []pixels[i];
    }
    for (int ch = 0; ch < 3; ++ch)
        noise[ch] /= len;
    // loop #2: do Wiener filter
    for (int idx = startIndex; idx < endIndex; ++idx)
    {
        int x = idx % img->width();
        int y = idx / img->width();
        auto offset = OFFSET(x, y);
        if (x < 1 || y < 1 || x >= img->width() - 1 || y >= img->height() - 1)
            continue;
        double rgb[3];
        QRgb pixelColor = img->pixel(x, y);
        int pixel[3] = {qRed(pixelColor),  qGreen(pixelColor), qBlue(pixelColor)};
        for (int ch = 0; ch < 3; ++ch)
        {
            rgb[ch] = pixel[ch] - mean[ch][offset];
            double t = variance[ch][offset] - noise[ch];
            if (t < 0.0)
                t = 0.0;
            variance[ch][offset] = fmax(variance[ch][offset], noise[ch]);
            rgb[ch] = rgb[ch] / variance[ch][offset] * t + mean[ch][offset];
        }
        img->setPixel(x, y, qRgb(rgb[0], rgb[1], rgb[2]));
    }
    for (int ch = 0; ch < 3; ++ch)
    {
        delete[] mean[ch];
        delete[] variance[ch];
    }
    return 0;
}

uint ImageProcessAlgorithm::BicubicRotate(ThreadParam *param)
{
    auto rp = (RotateParams*)(param->ctx);
    QImage *img = param->src; //目标图像
    QImage *src = rp->src;  //原图像
    const double sina = sin(rp->angle), cosa = cos(rp->angle);
    vec2<double> ncenter = { img->width() / 2.0, img->height() / 2.0 };
    vec2<double> ocenter = { src->width() / 2.0, src->height() / 2.0 };

    for (int i = param->startIndex; i < param->endIndex; ++i)
    {
        int x = i % img->width();
        int y = i / img->width();
        int xx = static_cast<int>(x - ncenter.x);
        int yy = static_cast<int>(y - ncenter.y);
        double oldx = xx * cosa - yy * sina + ocenter.x;
        double oldy = xx * sina + yy * cosa + ocenter.y;
        int iox = (int)oldx, ioy = (int)oldy;

        // out of interpolation border
        if (iox <= 1 || iox + 2 >= src->width() - 1 || ioy <= 1 || ioy + 2 >= src->height())
        {
            // but, still in the original image
            if (iox >= 0 && iox < src->width() && ioy>=0 && ioy < src->height())
                img->setPixel(x, y, src->pixel(iox, ioy));
            continue;
        }

        // Bicubic interpolation
        // 1st row
        auto p00 = getPixelHelp(src, iox - 1, ioy - 1);
        auto p10 = getPixelHelp(src, iox + 0, ioy - 1);
        auto p20 = getPixelHelp(src, iox + 1, ioy - 1);
        auto p30 = getPixelHelp(src, iox + 2, ioy - 1);

        // 2nd row
        auto p01 = getPixelHelp(src, iox - 1, ioy + 0);
        auto p11 = getPixelHelp(src, iox + 0, ioy + 0);
        auto p21 = getPixelHelp(src, iox + 1, ioy + 0);
        auto p31 = getPixelHelp(src, iox + 2, ioy + 0);

        // 3rd row
        auto p02 = getPixelHelp(src, iox - 1, ioy + 1);
        auto p12 = getPixelHelp(src, iox + 0, ioy + 1);
        auto p22 = getPixelHelp(src, iox + 1, ioy + 1);
        auto p32 = getPixelHelp(src, iox + 2, ioy + 1);

        // 4th row
        auto p03 = getPixelHelp(src, iox - 1, ioy + 2);
        auto p13 = getPixelHelp(src, iox + 0, ioy + 2);
        auto p23 = getPixelHelp(src, iox + 1, ioy + 2);
        auto p33 = getPixelHelp(src, iox + 2, ioy + 2);

        double result[3];
        for (int i = 0; i < 3; ++i)
        {
            double col0 = BicubicHermite(p00[i], p10[i], p20[i], p30[i], oldx - iox);
            double col1 = BicubicHermite(p01[i], p11[i], p21[i], p31[i], oldx - iox);
            double col2 = BicubicHermite(p02[i], p12[i], p22[i], p32[i], oldx - iox);
            double col3 = BicubicHermite(p03[i], p13[i], p23[i], p33[i], oldx - iox);
            result[i] = BicubicHermite(col0, col1, col2, col3, oldy - ioy);
            if (result[i] > 255.0)
                result[i] = 255.0;
            if (result[i] < 0.0)
                result[i] = 0.0;
        }
        img->setPixel(x, y, qRgb(result[0], result[1], result[2]));
    }

    return 0;
}

uint ImageProcessAlgorithm::BicubicScale(ThreadParam *param)
{
    auto sp = (ScaleParams*)(param->ctx);
    QImage *img = param->src; //目标图像
    QImage *src = sp->src; //原图像
    for (int i = param->startIndex; i < param->endIndex; ++i)
    {
        int ix = i % img->width();
        int iy = i / img->width();
        double x = ix / ((double)img->width() / src->width());
        double y = iy / ((double)img->height() / src->height());
        int fx = (int)x, fy = (int)y;

        // Handle the border
        if (fx - 1 <= 0 || fx + 2 >= src->width() - 1 || fy - 1 <= 0 || fy + 2 >= src->height() - 1)
        {
            fx = fx < 0 ? 0 : fx;
            fx = fx >= src->width() ? src->width() - 1 : fx;
            fy = fy < 0 ? 0 : fy;
            fy = fy >= src->height() ? src->height() - 1 : fy;
            img->setPixel(ix, iy, src->pixel(fx, fy));
            continue;
        }
        // Calc w
        double wx[4], wy[4];
        wx[0] = BicubicWeight(fx - 1 - x);
        wx[1] = BicubicWeight(fx + 0 - x);
        wx[2] = BicubicWeight(fx + 1 - x);
        wx[3] = BicubicWeight(fx + 2 - x);
        wy[0] = BicubicWeight(fy - 1 - y);
        wy[1] = BicubicWeight(fy + 0 - y);
        wy[2] = BicubicWeight(fy + 1 - y);
        wy[3] = BicubicWeight(fy + 2 - y);
        // Get pixels
        std::vector<int> p[4][4];
#define FILLPX(x, y, i, j) p[i][j]=getPixelHelp(src, x, y)
        FILLPX(fx - 1, fy + 0, 0, 1);
        FILLPX(fx - 1, fy + 1, 0, 2);
        FILLPX(fx - 1, fy + 2, 0, 3);
        FILLPX(fx + 0, fy - 1, 1, 0);
        FILLPX(fx + 0, fy + 0, 1, 1);
        FILLPX(fx + 0, fy + 1, 1, 2);
        FILLPX(fx + 0, fy + 2, 1, 3);
        FILLPX(fx + 1, fy - 1, 2, 0);
        FILLPX(fx + 1, fy + 0, 2, 1);
        FILLPX(fx + 1, fy + 1, 2, 2);
        FILLPX(fx + 1, fy + 2, 2, 3);
        FILLPX(fx + 2, fy - 1, 3, 0);
        FILLPX(fx + 2, fy + 0, 3, 1);
        FILLPX(fx + 2, fy + 1, 3, 2);
        FILLPX(fx + 2, fy + 2, 3, 3);
#undef FILLPX
        double rgb[3];
        rgb[0] = rgb[1] = rgb[2] = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                rgb[0] += p[i][j][0] * wx[i] * wy[j];
                rgb[1] += p[i][j][1] * wx[i] * wy[j];
                rgb[2] += p[i][j][2] * wx[i] * wy[j];
            }
        for (int i = 0; i < 3; ++i)
            rgb[i] = std::clamp(rgb[i], 0.0, 255.0);
        img->setPixel(ix, iy, qRgb(rgb[0], rgb[1], rgb[2]));
    }
    return 0;
}

