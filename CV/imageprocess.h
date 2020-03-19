#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include "imageprocessalgorithm.h"
#include <QObject>
#include <QThread>

class ImageProcessWorker : public QObject
{
    Q_OBJECT
public:
    typedef uint (*ImageProcessFunc)(ThreadParam *param);
    ImageProcessWorker();

public slots:
    void imageProcessByOpenMP(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByQThread(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByCUDA(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);
    void imageProcessByWinThread(QImage *image, ImageProcessFunc function, uint threadNumber = 5, bool loop = false);

signals:
    void processFinished(QImage *image);
private:
    int m_loopNumber;
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
