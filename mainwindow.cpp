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
    message(QMessageBox::Information, u8"提示", u8"正在处理，请稍等...", QMessageBox::NoButton, this)
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
    l_buttonGroup1->setExclusive(true);

    //test data
    ui->m_outputTextEdit->append(u8"初始化...");
    QString l_SourceDir = u8":/Icon/test.jpg";
    ui->m_inputPathLineEdit->setText(l_SourceDir);
    m_sourceImage = QImage(l_SourceDir).scaled(600, 400, Qt::KeepAspectRatio);
    ui->m_initPicture->setPixmap(QPixmap::fromImage(m_sourceImage));


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
        ui->m_outputTextEdit->append(u8"执行添加噪声噪声操作...");
        this->m_currentFunc = ImageProcessAlgorithm::saltAndPepperNoise;
        this->processImageHelp();
    });
    connect(ui->m_middleFilterPtn, &QPushButton::clicked, [=]{
        ui->m_outputTextEdit->append(u8"执行中值滤波操作...");
        if(ui->m_threadModeComboBox->currentIndex() != 2)
            this->m_currentFunc = ImageProcessAlgorithm::medianFilter;
        else
            this->m_currentFunc = ImageProcessAlgorithm::medianFilterCUDA;

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
            ui->m_outputTextEdit->append(QString(u8"执行旋转操作，旋转角度为: %1").arg(m_angle));
            m_angle = m_angle * std::acos(-1) / 180;
            if(ui->m_triangleCheckBox->isChecked())
                this->m_currentFunc = ImageProcessAlgorithm::BicubicRotate;
            if(ui->m_threadModeComboBox->currentIndex() == 2)
                this->m_currentFunc = ImageProcessAlgorithm::BicubicRotateCUDA;
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
            ui->m_outputTextEdit->append(QString(u8"执行放缩操作，放缩比例为: %1").arg(m_scale));
            if(ui->m_triangleCheckBox->isChecked())
                this->m_currentFunc = ImageProcessAlgorithm::BicubicScale;
            if(ui->m_threadModeComboBox->currentIndex() == 2)
                this->m_currentFunc = ImageProcessAlgorithm::BicubicScaleCUDA;
            this->processImageHelp();
        }
    });
    connect(ui->m_fourierToolButton, &QPushButton::clicked, [=]{
        auto returnButton = QMessageBox::information(this, u8"提示", u8"该方法复杂度较高，建议使用较小的图片尝试！",
                                                     QMessageBox::Yes | QMessageBox::Close);
        if(returnButton == QMessageBox::Yes)
        {
            ui->m_outputTextEdit->append(QString(u8"执行傅里叶变换，所需时长较长..."));
            this->m_currentFunc = ImageProcessAlgorithm::FourierTransform;
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
            ui->m_outputTextEdit->append(QString(u8"执行添加高斯噪声操作，均值为: %1，标准差为: %2").arg(m_mean).arg(m_stddev));
            this->m_currentFunc = ImageProcessAlgorithm::gaussianNoise;
            this->processImageHelp();
        }
    });
    connect(ui->m_smoothFilterPtn, &QPushButton::clicked, [=]{
        ui->m_outputTextEdit->append(QString(u8"执行线性平滑滤波操作..."));
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
            this->m_stddev = input.toDouble();
            ui->m_outputTextEdit->append(QString(u8"执行高斯滤波操作，标准差为: %1").arg(m_stddev));
            this->m_currentFunc = ImageProcessAlgorithm::gaussianFilter;
            this->processImageHelp();
        }
    });
    connect(ui->m_venusFilterPtn, &QPushButton::clicked, [=]{
        ui->m_outputTextEdit->append(QString(u8"执行维纳滤波操作..."));
        this->m_currentFunc = ImageProcessAlgorithm::WienerFilter;
        this->processImageHelp();
    });
    connect(ui->m_bilateralFilterPtn, &QPushButton::clicked, [=]{
        ui->m_outputTextEdit->append(QString(u8"执行双边滤波操作..."));
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
    m_sourceImage = QImage(l_SourceDir);
//    .scaled(600, 400, Qt::KeepAspectRatio);
    ui->m_initPicture->setPixmap(QPixmap::fromImage(m_sourceImage));
    ui->m_outputTextEdit->append(QString(u8"已添加图像，已自动缩放比例..."));
}

void MainWindow::handleResultImage(QImage *image)
{
    auto l_endTime = QDateTime::currentMSecsSinceEpoch();
    message.setText(u8"处理时间：" + QString::number(l_endTime - m_timeStart) + "ms.");
    if(!message.isVisible())
    {
        message.setVisible(true);
    }
    ui->m_processedPicture->setPixmap(QPixmap::fromImage(*image));
    //    ui->m_initPicture->setPixmap(QPixmap::fromImage((*m_sourceImage).scaled(600, 400, Qt::KeepAspectRatio)));
    m_processedImage = *image;
    ui->m_outputTextEdit->append(QString(u8"图像处理结束，处理时间为: %1ms.").arg(l_endTime - m_timeStart));
}

void MainWindow::processImageHelp()
{
    if(m_sourceImage.isNull())
    {
        QMessageBox::warning(this, u8"警告", u8"暂未选择图片！!");
        return;
    }
    bool isLoop = ui->m_isLoopCheckBox->isChecked();
    uint l_threadNumber = ui->m_threadNumberSpinBox->value();
    int l_threadMode = ui->m_threadModeComboBox->currentIndex();
    //确定输入图片
    if(ui->m_initPlotRadioButton->isChecked())
        m_toProcessImage = m_sourceImage.copy();
    else if(m_processedImage.isNull())
    {
        QMessageBox::warning(this, u8"警告", u8"暂无输出图片!");
        return;
    }
    else
        m_toProcessImage = m_processedImage;
    setParamHelp();

    //确定输出图片
    if(m_currentFunc == ImageProcessAlgorithm::BicubicScale || m_currentFunc == ImageProcessAlgorithm::BicubicScaleCUDA)
    {
        m_processedImage = QImage(m_toProcessImage.width() * m_scale, m_toProcessImage.height() * m_scale,
                                  m_toProcessImage.format());
        m_processedImage.fill(Qt::black);
    }
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicRotate || m_currentFunc == ImageProcessAlgorithm::BicubicRotateCUDA)
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
        m_processedImage = m_toProcessImage.copy();

    switch (l_threadMode)
    {
        case 0:
            emit processByOpenMP(&m_processedImage, m_currentFunc, l_threadNumber, isLoop);
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
    else if(m_currentFunc == ImageProcessAlgorithm::gaussianFilter)
        m_worker->setGaussianParam(m_mean, m_stddev);
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicScale || m_currentFunc == ImageProcessAlgorithm::BicubicScaleCUDA)
    {
        m_worker->setScaleParam(m_scale);
        m_worker->setSrcImage(&m_toProcessImage);
    }
    else if(m_currentFunc == ImageProcessAlgorithm::BicubicRotate || m_currentFunc == ImageProcessAlgorithm::BicubicRotateCUDA)
    {
        m_worker->setRotatearam(m_angle);
        m_worker->setSrcImage(&m_toProcessImage);
    }
    else if(m_currentFunc == ImageProcessAlgorithm::FourierTransform)
    {
        m_worker->setSrcImage(&m_toProcessImage);
    }
}

void MainWindow::on_m_loopNumberSpinBox_valueChanged(int arg1)
{
    m_worker->setLoopNumber(arg1);
    ui->m_outputTextEdit->append(QString(u8"已设置循环次数为: %1...").arg(arg1));
}

void MainWindow::on_m_isLoopCheckBox_stateChanged(int arg1)
{
    Q_UNUSED(arg1)
    ui->m_loopNumberSpinBox->setEnabled(ui->m_isLoopCheckBox->isChecked());
    ui->m_outputTextEdit->append(QString(u8"已设置循环处理，注意某些操作将耗时非常久.."));

}

void MainWindow::on_m_threadModeComboBox_currentIndexChanged(int index)
{
    if(index == 3 || index == 2)
        ui->m_threadNumberSpinBox->setEnabled(false);
    else if(!ui->m_threadNumberSpinBox->isEnabled())
        ui->m_threadNumberSpinBox->setEnabled(true);

    ui->m_outputTextEdit->append(QString(u8"已选择多线程处理方法为: %1...").arg(ui->m_threadModeComboBox->currentText()));
    if(index == 2)
    {
        ui->m_outputTextEdit->append(QString(u8"CUDA实现的方法仅有: %1，%2，%3，%4...").
                                     arg(u8"中值滤波").arg(u8"算术均值滤波").arg(u8"三阶插值旋转").arg(u8"三阶插值放缩"));
    }
}

ImageProcessAlgorithm::vec2<double> MainWindow::Vec2AfterRotate(
        const ImageProcessAlgorithm::vec2<double> &p, const double sa, const double ca)
{
    ImageProcessAlgorithm::vec2<double> ret;
    ret.x =  p.x * ca + p.y * sa;
    ret.y = -p.x * sa + p.y * ca;
    return ret;
}
