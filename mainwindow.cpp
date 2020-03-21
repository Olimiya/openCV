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
#include <QButtonGroup>
#include <QInputDialog>
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
    ui->m_loopNumberSpinBox->setEnabled(false);
    QButtonGroup *l_buttonGroup = new QButtonGroup(this);
    l_buttonGroup->addButton(ui->m_initPlotRadioButton);
    l_buttonGroup->addButton(ui->m_outputPlotRadioButton);
    l_buttonGroup->setExclusive(true);
    QButtonGroup *l_buttonGroup1 = new QButtonGroup(this);
    l_buttonGroup1->addButton(ui->m_triangleCheckBox);
    l_buttonGroup1->addButton(ui->m_fourierCheckBox);
    l_buttonGroup1->setExclusive(true);

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
    connect(this, &MainWindow::processBySingleThread, m_worker, &ImageProcessWorker::imageProcessBySingleThread);
    connect(this, &MainWindow::processByCUDA, m_worker, &ImageProcessWorker::imageProcessByCUDA);

    connect(m_worker, &ImageProcessWorker::processFinished, this, &MainWindow::handleResultImage);
    m_workThread->start();

    //connect
    connect(ui->m_saltNoisePth, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::saltAndPepperNoise;
        this->processImageHelp();
    });
    connect(ui->m_middleFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::medianFilter;
        this->processImageHelp();
    });
    connect(ui->m_rotatePtn, &QPushButton::clicked, [=]{
        bool ok;
        auto input = QInputDialog::getText(
                         this, u8"参数", u8"输入旋转角度：",QLineEdit::Normal, "0",
                         &ok, Qt::WindowFlags(), Qt::ImhDigitsOnly);
        if(ok && !input.isEmpty())
        {
            this->m_angle = input.toDouble();
            m_angle = m_angle * std::acos(-1) / 180;
            if(ui->m_triangleCheckBox->isChecked())
                this->m_currentFunc = ImageProcessAlgorithm::BicubicRotate;
            this->processImageHelp();
        }
    });
    connect(ui->m_scalePtn, &QPushButton::clicked, [=]{
        bool ok;
        auto input = QInputDialog::getText(
                         this, u8"参数", u8"输入缩放比例：",QLineEdit::Normal, "1",
                         &ok, Qt::WindowFlags(), Qt::ImhDigitsOnly);
        if(ok && !input.isEmpty())
        {
            this->m_scale = input.toDouble();
            if(ui->m_triangleCheckBox->isChecked())
                this->m_currentFunc = ImageProcessAlgorithm::BicubicScale;
            this->processImageHelp();
        }
    });
    connect(ui->m_gaussianNoisePtn, &QPushButton::clicked, [=]{
        bool ok;
        auto input = QInputDialog::getText(
                         this, u8"参数", u8"输入均值标准差(以,隔开)：",QLineEdit::Normal, "0,1",
                         &ok, Qt::WindowFlags(), Qt::ImhPreferNumbers);
        if(ok && !input.isEmpty())
        {
            auto params = input.split(",");
            if(params.size() != 2)
            {
                QMessageBox::warning(this, u8"警告", u8"输入错误!");
                return;
            }
            this->m_mean = params[0].toDouble();
            this->m_stddev = params[1].toDouble();
            this->m_currentFunc = ImageProcessAlgorithm::gaussianNoise;
            this->processImageHelp();
        }
    });
    connect(ui->m_smoothFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::ArithMeanFilter;
        this->processImageHelp();
    });
    connect(ui->m_gaussianFilterPtn, &QPushButton::clicked, [=]{
        bool ok;
        auto input = QInputDialog::getText(
                         this, u8"参数", u8"输入标准差：",QLineEdit::Normal, "1",
                         &ok, Qt::WindowFlags(), Qt::ImhDigitsOnly);
        if(ok && !input.isEmpty())
        {
            this->m_mean = input.toDouble();
            this->m_currentFunc = ImageProcessAlgorithm::gaussianFilter;
            this->processImageHelp();
        }
    });
    connect(ui->m_venusFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::WienerFilter;
        this->processImageHelp();
    });
    connect(ui->m_bilateralFilterPtn, &QPushButton::clicked, [=]{
        this->m_currentFunc = ImageProcessAlgorithm::saltAndPepperNoise;
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
    m_processedImage = *image;
}

void MainWindow::processImageHelp()
{
    if(m_sourceImage->isNull())
    {
        QMessageBox::warning(this, u8"警告", u8"暂未选择图片！!");
        return;
    }
    bool isLoop = ui->m_isLoopCheckBox->isChecked();
    uint l_threadNumber = ui->m_threadNumberSpinBox->value();
    int l_threadMode = ui->m_threadModeComboBox->currentIndex();
    //确定输入图片
    if(ui->m_initPlotRadioButton->isChecked())
        m_toProcessImage = m_sourceImage->copy();
    else if(m_processedImage.isNull())
    {
        QMessageBox::warning(this, u8"警告", u8"暂无输出图片!");
        return;
    }
    else
        m_toProcessImage = m_processedImage;
    setParamHelp();

    //确定输出图片
    if(m_currentFunc == ImageProcessAlgorithm::BicubicScale)
    {
        m_processedImage = QImage(m_toProcessImage.width() * m_scale, m_toProcessImage.height() * m_scale,
                                  m_toProcessImage.format());
        m_processedImage.fill(Qt::black);
    }
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicRotate)
    {
        // 计算旋转后的坐标
        double sina = sin(m_angle);
        double cosa = cos(m_angle);
        ImageProcessAlgorithm::vec2<double> lefttop = Vec2AfterRotate({ 0.0, 0.0 }, sina, cosa);
        ImageProcessAlgorithm::vec2<double> leftbottom = Vec2AfterRotate(
        { 0.0, (double)m_toProcessImage.height() - 1.0 }, sina, cosa);
        ImageProcessAlgorithm::vec2<double> righttop = Vec2AfterRotate(
        { (double)m_toProcessImage.width() - 1.0, 0.0 }, sina, cosa);
        ImageProcessAlgorithm::vec2<double> rightbottom = Vec2AfterRotate(
        { (double)m_toProcessImage.width() - 1.0, (double)m_toProcessImage.height() - 1.0 }, sina, cosa);

        double left   = std::min(lefttop.x, std::min(righttop.x, std::min(leftbottom.x, rightbottom.x)));
        double right  = std::max(lefttop.x, std::max(righttop.x, std::max(leftbottom.x, rightbottom.x)));
        double top    = std::min(lefttop.y, std::min(righttop.y, std::min(leftbottom.y, rightbottom.y)));
        double bottom = std::max(lefttop.y, std::max(righttop.y, std::max(leftbottom.y, rightbottom.y)));

        int width  = (int)abs(right - left) + 1;
        int height = (int)abs(top - bottom) + 1;
        m_processedImage = QImage(width, height, m_toProcessImage.format());
        m_processedImage.fill(Qt::black);
    }
    else
        m_processedImage = m_toProcessImage;

    switch (l_threadMode)
    {
        case 0:
            emit processByOpenMP(&m_toProcessImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 1:
            emit processByQThread(&m_processedImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 2:
            emit processByCUDA(&m_processedImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        case 3:
            emit processBySingleThread(&m_processedImage, m_currentFunc, l_threadNumber, isLoop);
            break;
        default:
            break;
    }
    m_timeStart = (QDateTime::currentMSecsSinceEpoch());
    message.setText(u8"正在处理，请稍等...");
    message.exec();
}

void MainWindow::setParamHelp()
{
    if(m_currentFunc == ImageProcessAlgorithm::gaussianNoise)
        m_worker->setGaussianParam(m_mean, m_stddev);
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicScale)
    {
        m_worker->setScaleParam(m_scale);
        m_worker->setSrcImage(&m_toProcessImage);
    }
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicRotate)
    {
        m_worker->setRotatearam(m_angle);
        m_worker->setSrcImage(&m_toProcessImage);
    }
}

void MainWindow::on_m_loopNumberSpinBox_valueChanged(int arg1)
{
    m_worker->setLoopNumber(arg1);
}

void MainWindow::on_m_isLoopCheckBox_stateChanged(int arg1)
{
    Q_UNUSED(arg1)
    ui->m_loopNumberSpinBox->setEnabled(ui->m_isLoopCheckBox->isChecked());
}

void MainWindow::on_m_threadModeComboBox_currentIndexChanged(int index)
{
    if(index == 3)
        ui->m_threadNumberSpinBox->setEnabled(false);
    else if(!ui->m_threadNumberSpinBox->isEnabled())
        ui->m_threadNumberSpinBox->setEnabled(true);
}

ImageProcessAlgorithm::vec2<double> MainWindow::Vec2AfterRotate(
        const ImageProcessAlgorithm::vec2<double> &p, const double sa, const double ca)
{
    ImageProcessAlgorithm::vec2<double> ret;
    ret.x =  p.x * ca + p.y * sa;
    ret.y = -p.x * sa + p.y * ca;
    return ret;
}
