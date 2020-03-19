#include "imageprocess.h"
#include <QDebug>
#include <omp.h>

class A
{};

void ImageProcessWorker::imageProcessByOpenMP(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{
    A *s = new A[5];
    int l_processNumber = 1;
    if(loop)
        l_processNumber = 100;

    while(l_processNumber--)
    {
        int m_nThreadNum = threadNumber;
        ThreadParam *m_pThreadParam = new ThreadParam[m_nThreadNum];
        int subLength = image->width() * image->height() / m_nThreadNum;
#pragma omp parallel for num_threads(m_nThreadNum)
        for (int i = 0; i < m_nThreadNum; ++i)
        {
            m_pThreadParam[i].startIndex = i * subLength;
            m_pThreadParam[i].endIndex = i != m_nThreadNum - 1 ?
                                                  (i + 1) * subLength - 1 : image->width() * image->height() - 1;
            m_pThreadParam[i].src = image;

            function(&m_pThreadParam[i]);
            qDebug() << omp_get_thread_num();
        }
        delete []m_pThreadParam;
    }
    emit processFinished(image);
}

void ImageProcessWorker::imageProcessByQThread(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{
    int l_processNumber = 1;
    if(loop)
        l_processNumber = 100;

    int l_processFinshedNum = 0;
    while(l_processNumber--)
    {
        int m_nThreadNum = threadNumber;
        ThreadParam *m_pThreadParam = new ThreadParam[m_nThreadNum];
        int subLength = image->width() * image->height() / m_nThreadNum;
        int taskFinishedNum = 0;
        std::vector<QtProcessThread*> l_threads;
        for (int i = 0; i < m_nThreadNum; ++i)
        {
            m_pThreadParam[i].startIndex = i * subLength;
            m_pThreadParam[i].endIndex = i != m_nThreadNum - 1 ?
                                                  (i + 1) * subLength - 1 : image->width() * image->height() - 1;
            m_pThreadParam[i].src = image;

            QtProcessThread *thread= new QtProcessThread(&m_pThreadParam[i], function);
            connect(thread, &QThread::finished, thread, &QThread::deleteLater);
            connect(thread, &QThread::finished, thread, [&]{
                taskFinishedNum++;
                if(taskFinishedNum == m_nThreadNum)
                    delete []m_pThreadParam;
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

}

void ImageProcessWorker::imageProcessByWinThread(QImage *image, ImageProcessFunc function, uint threadNumber, bool loop)
{

}

ImageProcessWorker::ImageProcessWorker():m_loopNumber(100){}


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
