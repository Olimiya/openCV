#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#include <QLabel>
#include <QVBoxLayout>
#include <QFile>
#include <QDir>
#include <QFileDialog>
#include <QDebug>
#include <QDateTime>
#include <omp.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    message(QMessageBox::Information, u8"提示", u8"正在处理，请稍等...", QMessageBox::NoButton, this),
    m_sourceImage(nullptr)
{
    ui->setupUi(this);


    QFile qss(":/QSS/TotalQSS.qss");
    if(qss.open(QFile::ReadOnly))
    {
        QString content = qss.readAll();
        this->setStyleSheet(content);
    }
    message.setStyleSheet("");

    //test data
    QString l_SourceDir = "G:/photo/1/11.jpg";
    ui->m_inputPathLineEdit->setText(l_SourceDir);
    m_sourceImage = new QImage(l_SourceDir);
    ui->m_initPicture->setPixmap(QPixmap::fromImage((*m_sourceImage).scaled(600, 400, Qt::KeepAspectRatio)));


    //多线程的处理对象
    m_worker = new ImageProcessWorker;
    m_workThread = new QThread;
    m_worker->moveToThread(m_workThread);
    connect(m_workThread, &QThread::finished, m_worker, &QObject::deleteLater);
    connect(m_workThread, &QThread::finished, m_workThread, &QThread::deleteLater);

    qRegisterMetaType<ImageProcessWorker::ImageProcessFunc>("ImageProcessWorker::ImageProcessFunc");
    connect(this, &MainWindow::processByOpenMP, m_worker, &ImageProcessWorker::imageProcessByOpenMP);
    connect(this, &MainWindow::processByQThread, m_worker, &ImageProcessWorker::imageProcessByQThread);
    connect(this, &MainWindow::processByWinThread, m_worker, &ImageProcessWorker::imageProcessByWinThread);
    connect(this, &MainWindow::processByCUDA, m_worker, &ImageProcessWorker::imageProcessByCUDA);

    connect(m_worker, &ImageProcessWorker::processFinished, this, &MainWindow::handleResultImage);
    m_workThread->start();

    //connect
    connect(ui->m_saltNoisePth, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_middleFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::medianFilter;
        this->processImageHelp();
    });
    connect(ui->m_rotatePtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_scalePtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_gaussianNoisePtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_smoothFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_gaussianFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
    connect(ui->m_bilateralFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::addNoise;
        this->processImageHelp();
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_m_filepathButton_clicked()
{
    QString workingPath;
    if (!ui->m_inputPathLineEdit->text().isEmpty())
        workingPath = ui->m_inputPathLineEdit->text();
    else
        workingPath = QDir::currentPath();
    QString l_SourceDir = QFileDialog::getOpenFileName(
                              this, QString(u8"选择图片"),
                              workingPath, "Images (*.png *.jpg *bmp)");
    if (l_SourceDir.isEmpty())
        return;
    ui->m_inputPathLineEdit->setText(l_SourceDir);
    m_sourceImage = new QImage(l_SourceDir);
    ui->m_initPicture->setPixmap(QPixmap::fromImage((*m_sourceImage).scaled(600, 400, Qt::KeepAspectRatio)));
}

void MainWindow::handleResultImage(QImage *image)
{
    auto l_endTime = QDateTime::currentMSecsSinceEpoch();
    message.setText(u8"处理时间：" + QString::number(l_endTime - m_timeStart) + "ms.");
    if(!message.isVisible())
    {
        message.setVisible(true);
    }
    ui->m_processedPicture->setPixmap(QPixmap::fromImage(image->scaled(600, 400, Qt::KeepAspectRatio)));
    ui->m_initPicture->setPixmap(QPixmap::fromImage((*m_sourceImage).scaled(600, 400, Qt::KeepAspectRatio)));
}

void MainWindow::processImageHelp()
{
    bool isLoop = ui->m_isLoopCheckBox->isChecked();
    uint l_threadNumber = ui->m_threadNumberSpinBox->value();
    int l_threadMode = ui->m_threadModeComboBox->currentIndex();
    m_toProcessImage = m_sourceImage->copy();

    switch (l_threadMode)
    {
        case 0:
            emit processByOpenMP(&m_toProcessImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 1:
            emit processByQThread(&m_toProcessImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 2:
            emit processByWinThread(&m_toProcessImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 3:
            emit processByCUDA(&m_toProcessImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        default:
            break;
    }
    m_timeStart = (QDateTime::currentMSecsSinceEpoch());
    message.setText(u8"正在处理，请稍等...");
    message.exec();
}
