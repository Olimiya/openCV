/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout_2;
    QTabWidget *m_tabwidget;
    QWidget *m_funtionTab;
    QHBoxLayout *horizontalLayout;
    QToolButton *m_saltNoisePth;
    QToolButton *m_middleFilterPtn;
    QFrame *line;
    QToolButton *m_rotatePtn;
    QToolButton *m_scalePtn;
    QCheckBox *m_triangleCheckBox;
    QCheckBox *m_fourierCheckBox;
    QFrame *line_2;
    QToolButton *m_gaussianNoisePtn;
    QToolButton *m_smoothFilterPtn;
    QToolButton *m_gaussianFilterPtn;
    QToolButton *m_venusFilterPtn;
    QFrame *line_3;
    QToolButton *m_bilateralFilterPtn;
    QSpacerItem *horizontalSpacer_2;
    QWidget *tab_2;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QComboBox *m_threadModeComboBox;
    QLabel *label_2;
    QSpinBox *m_threadNumberSpinBox;
    QFrame *line_4;
    QCheckBox *m_isLoopCheckBox;
    QSpacerItem *horizontalSpacer;
    QGridLayout *gridLayout;
    QLabel *m_initPicture;
    QLabel *m_processedPicture;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_4;
    QLineEdit *m_inputPathLineEdit;
    QPushButton *m_filepathButton;
    QTextBrowser *m_outputTextBrowser;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(920, 661);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        gridLayout_2 = new QGridLayout(centralWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(-1, 0, -1, -1);
        m_tabwidget = new QTabWidget(centralWidget);
        m_tabwidget->setObjectName(QString::fromUtf8("m_tabwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_tabwidget->sizePolicy().hasHeightForWidth());
        m_tabwidget->setSizePolicy(sizePolicy);
        m_funtionTab = new QWidget();
        m_funtionTab->setObjectName(QString::fromUtf8("m_funtionTab"));
        horizontalLayout = new QHBoxLayout(m_funtionTab);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        m_saltNoisePth = new QToolButton(m_funtionTab);
        m_saltNoisePth->setObjectName(QString::fromUtf8("m_saltNoisePth"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/Icon/salt.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_saltNoisePth->setIcon(icon);
        m_saltNoisePth->setIconSize(QSize(30, 30));
        m_saltNoisePth->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_saltNoisePth->setAutoRaise(true);

        horizontalLayout->addWidget(m_saltNoisePth);

        m_middleFilterPtn = new QToolButton(m_funtionTab);
        m_middleFilterPtn->setObjectName(QString::fromUtf8("m_middleFilterPtn"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/Icon/mid.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_middleFilterPtn->setIcon(icon1);
        m_middleFilterPtn->setIconSize(QSize(30, 30));
        m_middleFilterPtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_middleFilterPtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_middleFilterPtn);

        line = new QFrame(m_funtionTab);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

        m_rotatePtn = new QToolButton(m_funtionTab);
        m_rotatePtn->setObjectName(QString::fromUtf8("m_rotatePtn"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/Icon/rotate.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_rotatePtn->setIcon(icon2);
        m_rotatePtn->setIconSize(QSize(30, 30));
        m_rotatePtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_rotatePtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_rotatePtn);

        m_scalePtn = new QToolButton(m_funtionTab);
        m_scalePtn->setObjectName(QString::fromUtf8("m_scalePtn"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/Icon/scale.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_scalePtn->setIcon(icon3);
        m_scalePtn->setIconSize(QSize(30, 30));
        m_scalePtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_scalePtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_scalePtn);

        m_triangleCheckBox = new QCheckBox(m_funtionTab);
        m_triangleCheckBox->setObjectName(QString::fromUtf8("m_triangleCheckBox"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8(":/Icon/triangle.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_triangleCheckBox->setIcon(icon4);
        m_triangleCheckBox->setChecked(true);

        horizontalLayout->addWidget(m_triangleCheckBox);

        m_fourierCheckBox = new QCheckBox(m_funtionTab);
        m_fourierCheckBox->setObjectName(QString::fromUtf8("m_fourierCheckBox"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8(":/Icon/fourier.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_fourierCheckBox->setIcon(icon5);

        horizontalLayout->addWidget(m_fourierCheckBox);

        line_2 = new QFrame(m_funtionTab);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_2);

        m_gaussianNoisePtn = new QToolButton(m_funtionTab);
        m_gaussianNoisePtn->setObjectName(QString::fromUtf8("m_gaussianNoisePtn"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8(":/Icon/noise.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_gaussianNoisePtn->setIcon(icon6);
        m_gaussianNoisePtn->setIconSize(QSize(30, 30));
        m_gaussianNoisePtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_gaussianNoisePtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_gaussianNoisePtn);

        m_smoothFilterPtn = new QToolButton(m_funtionTab);
        m_smoothFilterPtn->setObjectName(QString::fromUtf8("m_smoothFilterPtn"));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8(":/Icon/filter.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_smoothFilterPtn->setIcon(icon7);
        m_smoothFilterPtn->setIconSize(QSize(30, 30));
        m_smoothFilterPtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_smoothFilterPtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_smoothFilterPtn);

        m_gaussianFilterPtn = new QToolButton(m_funtionTab);
        m_gaussianFilterPtn->setObjectName(QString::fromUtf8("m_gaussianFilterPtn"));
        m_gaussianFilterPtn->setIcon(icon7);
        m_gaussianFilterPtn->setIconSize(QSize(30, 30));
        m_gaussianFilterPtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_gaussianFilterPtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_gaussianFilterPtn);

        m_venusFilterPtn = new QToolButton(m_funtionTab);
        m_venusFilterPtn->setObjectName(QString::fromUtf8("m_venusFilterPtn"));
        m_venusFilterPtn->setIcon(icon7);
        m_venusFilterPtn->setIconSize(QSize(30, 30));
        m_venusFilterPtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_venusFilterPtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_venusFilterPtn);

        line_3 = new QFrame(m_funtionTab);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setFrameShape(QFrame::VLine);
        line_3->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_3);

        m_bilateralFilterPtn = new QToolButton(m_funtionTab);
        m_bilateralFilterPtn->setObjectName(QString::fromUtf8("m_bilateralFilterPtn"));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8(":/Icon/double.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_bilateralFilterPtn->setIcon(icon8);
        m_bilateralFilterPtn->setIconSize(QSize(30, 30));
        m_bilateralFilterPtn->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        m_bilateralFilterPtn->setAutoRaise(true);

        horizontalLayout->addWidget(m_bilateralFilterPtn);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        m_tabwidget->addTab(m_funtionTab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        horizontalLayout_2 = new QHBoxLayout(tab_2);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(tab_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        m_threadModeComboBox = new QComboBox(tab_2);
        m_threadModeComboBox->addItem(QString());
        m_threadModeComboBox->addItem(QString());
        m_threadModeComboBox->addItem(QString());
        m_threadModeComboBox->addItem(QString());
        m_threadModeComboBox->setObjectName(QString::fromUtf8("m_threadModeComboBox"));

        horizontalLayout_2->addWidget(m_threadModeComboBox);

        label_2 = new QLabel(tab_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_2->addWidget(label_2);

        m_threadNumberSpinBox = new QSpinBox(tab_2);
        m_threadNumberSpinBox->setObjectName(QString::fromUtf8("m_threadNumberSpinBox"));
        m_threadNumberSpinBox->setMaximum(15);
        m_threadNumberSpinBox->setValue(5);

        horizontalLayout_2->addWidget(m_threadNumberSpinBox);

        line_4 = new QFrame(tab_2);
        line_4->setObjectName(QString::fromUtf8("line_4"));
        line_4->setFrameShape(QFrame::VLine);
        line_4->setFrameShadow(QFrame::Sunken);

        horizontalLayout_2->addWidget(line_4);

        m_isLoopCheckBox = new QCheckBox(tab_2);
        m_isLoopCheckBox->setObjectName(QString::fromUtf8("m_isLoopCheckBox"));

        horizontalLayout_2->addWidget(m_isLoopCheckBox);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        m_tabwidget->addTab(tab_2, QString());

        gridLayout_2->addWidget(m_tabwidget, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        m_initPicture = new QLabel(centralWidget);
        m_initPicture->setObjectName(QString::fromUtf8("m_initPicture"));
        m_initPicture->setAlignment(Qt::AlignCenter);
        m_initPicture->setMargin(6);

        gridLayout->addWidget(m_initPicture, 1, 0, 1, 1);

        m_processedPicture = new QLabel(centralWidget);
        m_processedPicture->setObjectName(QString::fromUtf8("m_processedPicture"));
        m_processedPicture->setAlignment(Qt::AlignCenter);
        m_processedPicture->setMargin(6);

        gridLayout->addWidget(m_processedPicture, 1, 1, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_3->addWidget(label_4);

        m_inputPathLineEdit = new QLineEdit(centralWidget);
        m_inputPathLineEdit->setObjectName(QString::fromUtf8("m_inputPathLineEdit"));

        horizontalLayout_3->addWidget(m_inputPathLineEdit);

        m_filepathButton = new QPushButton(centralWidget);
        m_filepathButton->setObjectName(QString::fromUtf8("m_filepathButton"));
        m_filepathButton->setMinimumSize(QSize(25, 30));
        m_filepathButton->setMaximumSize(QSize(25, 30));
        QIcon icon9;
        icon9.addFile(QString::fromUtf8(":/Icon/search.png"), QSize(), QIcon::Normal, QIcon::Off);
        m_filepathButton->setIcon(icon9);
        m_filepathButton->setIconSize(QSize(25, 30));
        m_filepathButton->setFlat(true);

        horizontalLayout_3->addWidget(m_filepathButton);


        gridLayout->addLayout(horizontalLayout_3, 0, 0, 1, 2);

        gridLayout->setRowStretch(0, 1);
        gridLayout->setRowStretch(1, 5);
        gridLayout->setColumnStretch(0, 1);
        gridLayout->setColumnStretch(1, 1);

        gridLayout_2->addLayout(gridLayout, 1, 0, 1, 1);

        m_outputTextBrowser = new QTextBrowser(centralWidget);
        m_outputTextBrowser->setObjectName(QString::fromUtf8("m_outputTextBrowser"));

        gridLayout_2->addWidget(m_outputTextBrowser, 2, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 920, 25));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);

        retranslateUi(MainWindow);

        m_tabwidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        m_saltNoisePth->setText(QApplication::translate("MainWindow", "\346\244\222\347\233\220\345\231\252\345\243\260", nullptr));
        m_middleFilterPtn->setText(QApplication::translate("MainWindow", "\344\270\255\345\200\274\346\273\244\346\263\242", nullptr));
        m_rotatePtn->setText(QApplication::translate("MainWindow", " \346\227\213\350\275\254", nullptr));
        m_scalePtn->setText(QApplication::translate("MainWindow", "\347\274\251\346\224\276", nullptr));
        m_triangleCheckBox->setText(QApplication::translate("MainWindow", "\344\270\211\351\230\266\346\217\222\345\200\274", nullptr));
        m_fourierCheckBox->setText(QApplication::translate("MainWindow", "\345\202\205\351\207\214\345\217\266\345\217\230\346\215\242", nullptr));
        m_gaussianNoisePtn->setText(QApplication::translate("MainWindow", "\351\253\230\346\226\257\345\231\252\345\243\260", nullptr));
        m_smoothFilterPtn->setText(QApplication::translate("MainWindow", "\345\271\263\346\273\221\347\272\277\346\200\247\346\273\244\346\263\242", nullptr));
        m_gaussianFilterPtn->setText(QApplication::translate("MainWindow", "\351\253\230\346\226\257\346\273\244\346\263\242", nullptr));
        m_venusFilterPtn->setText(QApplication::translate("MainWindow", "\347\273\264\347\272\263\346\273\244\346\263\242", nullptr));
        m_bilateralFilterPtn->setText(QApplication::translate("MainWindow", "\345\217\214\350\276\271\346\273\244\346\263\242\345\231\250", nullptr));
        m_tabwidget->setTabText(m_tabwidget->indexOf(m_funtionTab), QApplication::translate("MainWindow", "\345\212\237\350\203\275", nullptr));
        label->setText(QApplication::translate("MainWindow", " \345\244\232\347\272\277\347\250\213\357\274\232", nullptr));
        m_threadModeComboBox->setItemText(0, QApplication::translate("MainWindow", "OpenMP", nullptr));
        m_threadModeComboBox->setItemText(1, QApplication::translate("MainWindow", "QtThread", nullptr));
        m_threadModeComboBox->setItemText(2, QApplication::translate("MainWindow", "Win\345\244\232\347\272\277\347\250\213", nullptr));
        m_threadModeComboBox->setItemText(3, QApplication::translate("MainWindow", "CUDA", nullptr));

        label_2->setText(QApplication::translate("MainWindow", " \347\272\277\347\250\213\346\225\260\351\207\217", nullptr));
        m_isLoopCheckBox->setText(QApplication::translate("MainWindow", "\345\276\252\347\216\257\345\244\204\347\220\206100\346\254\241", nullptr));
        m_tabwidget->setTabText(m_tabwidget->indexOf(tab_2), QApplication::translate("MainWindow", "\345\244\204\347\220\206\346\226\271\345\274\217", nullptr));
        m_initPicture->setText(QApplication::translate("MainWindow", "\350\276\223\345\205\245", nullptr));
        m_processedPicture->setText(QApplication::translate("MainWindow", "\350\276\223\345\207\272", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "\351\200\211\346\213\251\345\234\260\345\235\200\357\274\232", nullptr));
        m_filepathButton->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
