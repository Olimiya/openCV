#include "imageprocess.h"
#include <QDebug>
#include <omp.h>

void ImageProcessWorker::imageProcessByOpenMP(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{
    int l_processNumber = 1;
    if(loop)
        l_processNumber = m_loopNumber;

    while(l_processNumber--)
    {
        int m_nThreadNum = threadNumber;
        ThreadParam *m_pThreadParam = wrapParamHelp(image, function, threadNumber);
#pragma omp parallel for num_threads(m_nThreadNum)
        for (int i = 0; i < m_nThreadNum; ++i)
        {
            function(&m_pThreadParam[i]);
            qDebug() << omp_get_thread_num();
        }
        deleteParamHelp(m_pThreadParam, function, m_nThreadNum);
    }
    emit processFinished(image);
}

void ImageProcessWorker::imageProcessByQThread(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{
    int l_processNumber = 1;
    if(loop)
        l_processNumber = m_loopNumber;

    while(l_processNumber--)
    {
        int m_nThreadNum = threadNumber;
        ThreadParam *m_pThreadParam = wrapParamHelp(image, function, threadNumber);
        int taskFinishedNum = 0;
        std::vector<QtProcessThread*> l_threads;
        for (int i = 0; i < m_nThreadNum; ++i)
        {
            QtProcessThread *thread= new QtProcessThread(&m_pThreadParam[i], function);
            connect(thread, &QThread::finished, thread, &QThread::deleteLater);
            connect(thread, &QThread::finished, thread, [&]{
                taskFinishedNum++;
                if(taskFinishedNum == m_nThreadNum)
                    this->deleteParamHelp(m_pThreadParam, function, m_nThreadNum);
            });
            l_threads.push_back(thread);
            thread->start();
        }
        for(auto i : l_threads)
            i->wait();
    }
    emit processFinished(image);
}

void ImageProcessWorker::imageProcessByCUDA(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{
    //该方法跟单线程完全一样
    //因为对于CPU就是单线程，多线程是在function中调用GPU进行并行计算
    imageProcessBySingleThread(image, function, threadNumber, loop);
}

void ImageProcessWorker::imageProcessBySingleThread(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop)
{
    Q_UNUSED(threadNumber)
    int l_processNumber = 1;
    if(loop)
        l_processNumber = m_loopNumber;
    while(l_processNumber--)
    {
        ThreadParam *param = wrapParamHelp(image, function, 1);
        function(param);
        deleteParamHelp(param, function, 1);
    }
    emit processFinished(image);
}

ThreadParam *ImageProcessWorker::wrapParamHelp(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber)
{
    int m_nThreadNum = threadNumber;
    ThreadParam *m_pThreadParam = new ThreadParam[m_nThreadNum];
    int subLength = image->width() * image->height() / m_nThreadNum;
    for (int i = 0; i < m_nThreadNum; ++i)
    {
        m_pThreadParam[i].startIndex = i * subLength;
        m_pThreadParam[i].endIndex = i != m_nThreadNum - 1 ?
                                              (i + 1) * subLength - 1 : image->width() * image->height() - 1;
        if(function == ImageProcessAlgorithm::medianFilter)
            m_pThreadParam[i].maxSpan = MAX_SPAN;
        else if(function == ImageProcessAlgorithm::gaussianNoise || function == ImageProcessAlgorithm::gaussianFilter)
        {
            ImageProcessAlgorithm::GassuianParam *ctr = new ImageProcessAlgorithm::GassuianParam;
            ctr->mean = m_mean;
            ctr->stddev = m_stddev;
            m_pThreadParam[i].ctx = ctr;
        }
        else if(function == ImageProcessAlgorithm::BicubicScale || function == ImageProcessAlgorithm::BicubicScaleCUDA)
        {
            ImageProcessAlgorithm::ScaleParams *ctr = new ImageProcessAlgorithm::ScaleParams;
            ctr->scale = m_scale;
            ctr->src = m_srcImage;
            m_pThreadParam[i].ctx = ctr;
        }
        else if(function == ImageProcessAlgorithm::BicubicRotate || function == ImageProcessAlgorithm::BicubicRotateCUDA)
        {
            ImageProcessAlgorithm::RotateParams *ctr = new ImageProcessAlgorithm::RotateParams;
            ctr->angle = m_angle;
            ctr->src = m_srcImage;
            m_pThreadParam[i].ctx = ctr;
        }
        else if(function == ImageProcessAlgorithm::FourierTransform)
        {
            m_pThreadParam[i].ctx = m_srcImage;
        }

        m_pThreadParam[i].src = image;
    }
    return m_pThreadParam;
}

void ImageProcessWorker::deleteParamHelp(ThreadParam *param, ImageProcessWorker::ImageProcessFunc function, uint threadNumber)
{
    if(function == ImageProcessAlgorithm::gaussianNoise || function == ImageProcessAlgorithm::gaussianFilter
            || function == ImageProcessAlgorithm::BicubicRotate || function == ImageProcessAlgorithm::BicubicScale
            || function == ImageProcessAlgorithm::BicubicScaleCUDA || function == ImageProcessAlgorithm::BicubicRotateCUDA)
    {
        for(uint i = 0; i < threadNumber; i++)
        {
            auto gp = (ImageProcessAlgorithm::GassuianParam*)(param[i].ctx);
            delete gp;
        }
    }
    delete []param;
}

ImageProcessWorker::ImageProcessWorker():m_loopNumber(100), m_mean(0), m_stddev(1), m_scale(1), m_angle(0),m_srcImage(){}

void ImageProcessWorker::setLoopNumber(int loopNumber)
{
    m_loopNumber = loopNumber;
}

void ImageProcessWorker::setGaussianParam(double mean, double stddev)
{
    m_mean = mean;
    m_stddev = stddev;
}
void ImageProcessWorker::setScaleParam(double scale)
{
    m_scale = scale;
}
void ImageProcessWorker::setRotatearam(double angle)
{
    m_angle = angle;
}
void ImageProcessWorker::setSrcImage(QImage *image)
{
    m_srcImage = image;
}

QtProcessThread::QtProcessThread(ThreadParam *param, ImageProcessWorker::ImageProcessFunc function)
    :m_param(param), m_function(function){}

void QtProcessThread::run()
{
    if(!isInterruptionRequested())
    {
        m_function(m_param);
        qDebug() << this->currentThreadId();
    }
}
