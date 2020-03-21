#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include "imageprocessalgorithm.h"
#include <QObject>
#include <QThread>
#define MAX_SPAN 3

class ImageProcessWorker : public QObject
{
    Q_OBJECT
public:
    typedef uint (*ImageProcessFunc)(ThreadParam *param);
    ImageProcessWorker();
    void setLoopNumber(int loopNumber);
    void setGaussianParam(double mean, double stddev);
    void setScaleParam(double scale);
    void setRotatearam(double angle);
    void setSrcImage(QImage *image);

public slots:
    void imageProcessBySingleThread(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByOpenMP(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByQThread(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByCUDA(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);

signals:
    void processFinished(QImage *image);
private:
    int m_loopNumber;
    //Gaussian
    double m_mean;
    double m_stddev;
    //Rotate Scale
    double m_scale;
    double m_angle;
    QImage *m_srcImage;

    ThreadParam *wrapParamHelp(QImage *image, ImageProcessFunc function, uint threadNumber = 5);
    void deleteParamHelp(ThreadParam *param, ImageProcessFunc function, uint threadNumber = 5);
};

class QtProcessThread : public QThread
{
public:
    QtProcessThread();
    QtProcessThread(ThreadParam *param, ImageProcessWorker::ImageProcessFunc function);
    ThreadParam *m_param;
    ImageProcessWorker::ImageProcessFunc m_function;
protected:
    virtual void run() override;
};

#endif // IMAGEPROCESS_H
