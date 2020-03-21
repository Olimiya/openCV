#-------------------------------------------------
#
# Project created by QtCreator 2020-03-11T12:50:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = openCV
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++17

# OpenCV
INCLUDEPATH += $$(OPENCV_CUDA)/include

LIBS += -L$$(OPENCV_CUDA)/x64/vc15/lib \
        -lopencv_aruco420d -lopencv_bgsegm420d  -lopencv_bioinspired420d  -lopencv_calib3d420d \
        -lopencv_ccalib420d  -lopencv_core420d  -lopencv_cudaarithm420d  -lopencv_cudabgsegm420d \
        -lopencv_cudacodec420d  -lopencv_cudafilters420d  -lopencv_cudaimgproc420d  -lopencv_cudalegacy420d \
        -lopencv_cudastereo420d  -lopencv_cudev420d  -lopencv_datasets420d  -lopencv_dnn420d \
        -lopencv_dnn_objdetect420d  -lopencv_dnn_superres420d  -lopencv_dpm420d  -lopencv_face420d \
        -lopencv_features2d420d  -lopencv_flann420d  -lopencv_fuzzy420d  -lopencv_gapi420d \
        -lopencv_hfs420d  -lopencv_highgui420d  -lopencv_imgcodecs420d  -lopencv_imgproc420d \
        -lopencv_img_hash420d  -lopencv_line_descriptor420d  -lopencv_ml420d  -lopencv_objdetect420d \
        -lopencv_optflow420d  -lopencv_phase_unwrapping420d  -lopencv_photo420d -lopencv_plot420d \
        -lopencv_quality420d -lopencv_reg420d -lopencv_rgbd420d -lopencv_saliency420d -lopencv_shape420d \
        -lopencv_stereo420d -lopencv_structured_light420d -lopencv_surface_matching420d -lopencv_text420d \
        -lopencv_tracking420d -lopencv_video420d -lopencv_videoio420d -lopencv_xfeatures2d420d -lopencv_ximgproc420d\
        -lopencv_xobjdetect420d -lopencv_xphoto420d -lopencv_cudaobjdetect420d -lopencv_cudaoptflow420d \
        -lopencv_cudawarping420d -lopencv_videostab420d -lopencv_superres420d -lopencv_stitching420d \
        -lopencv_cudafeatures2d420d

# CUDA
INCLUDEPATH += $$(CUDA_PATH)/include \
        F:\tools\CUDA10.2\Samples\common\inc
LIBS += -L$$(CUDA_PATH)/lib/x64 \
       -lcublas -lcublasLt -lcuda -lcudadevrt -lcudart -lcudart_static -lcufft -lcufftw -lcurand -lcusolver \
       -lcusolverMg -lcusparse -lnppc -lnppial -lnppicc -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppist \
       -lnppisu -lnppitc -lnpps -lnvblas -lnvgraph -lnvjpeg -lnvml -lnvrtc -lOpenCL

SYSTEM_NAME = x64 # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64# '32' or '64', depending on your system
CUDA_ARCH = compute_60  # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math # default setting

CUDA_INC += $$join(INCLUDEPATH,'" -I"','-I"','"')

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CUDA_SOURCES += CV/kernel.cu \
            CV/medianFilter.cu

# CUDA_PATH = F:/tools/CUDA10.2/CUDA_Development
CONFIG(debug, debug|release){
    cuda_d.input = CUDA_SOURCES
    cuda_d.output   = ${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$(CUDA_PATH)/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
    --compile -cudart static -g -DWIN32 -D_MBCS \
    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
    -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
# Release settings
    cuda.input= CUDA_SOURCES
    cuda.output   = ${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $$CUDA_PATH/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
    --compile -cudart static -DWIN32 -D_MBCS \
    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    CV/imageprocess.cpp \
    CV/imageprocessalgorithm.cpp
HEADERS += \
        mainwindow.h \
    CV/imageprocess.h \
    CV/imageprocessalgorithm.h
FORMS += \
        mainwindow.ui

QMAKE_CXXFLAGS+=/openmp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    opencv.qrc
