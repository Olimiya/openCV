#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QMessageBox>
#include "CV/imageprocess.h"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void handleResultImage(QImage *image);

    void on_m_filepathButton_clicked();
    void on_m_loopNumberSpinBox_valueChanged(int arg1);
    void on_m_isLoopCheckBox_stateChanged(int arg1);
    void on_m_threadModeComboBox_currentIndexChanged(int index);

signals:
    void processByOpenMP(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processByQThread(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processBySingleThread(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processByCUDA(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);

private:
    Ui::MainWindow *ui;
    QMessageBox message;
    QImage *m_sourceImage;
    QImage m_toProcessImage;
    QImage m_processedImage;

    qint64 m_timeStart;
    ImageProcessWorker::ImageProcessFunc m_currentFunc;
    //param
    double m_mean;
    double m_stddev;
    double m_scale;
    double m_angle;

    ImageProcessWorker *m_worker;
    QThread *m_workThread;
    void processImageHelp();

    void setParamHelp();

    inline ImageProcessAlgorithm::vec2<double> Vec2AfterRotate(
            const ImageProcessAlgorithm::vec2<double> &p, const double sa, const double ca);
};

#endif // MAINWINDOW_H
