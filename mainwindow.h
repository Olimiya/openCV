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
    void on_m_filepathButton_clicked();

    void handleResultImage(QImage *image);

signals:
    void processByOpenMP(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processByQThread(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processByWinThread(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);
    void processByCUDA(QImage *image, ImageProcessWorker::ImageProcessFunc function, uint threadNumber, bool loop);

private:
    Ui::MainWindow *ui;
    QMessageBox message;
    QImage *m_sourceImage;
    QImage m_toProcessImage;
    qint64 m_timeStart;
    ImageProcessWorker::ImageProcessFunc m_currentFunc;

    ImageProcessWorker *m_worker;
    QThread *m_workThread;
    void processImageHelp();
};

#endif // MAINWINDOW_H
