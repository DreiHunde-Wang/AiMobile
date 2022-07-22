// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "scrfd.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include<unistd.h>

#include "ast_rawosd.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

int SnapFlag  = 0;
static int g_modelid =-1 ;

//char  szSaveRoiPic [256] ={0};
char  szSaveRoiPic [1024] ={0};
static int write_img(cv::Mat &rgb)
{
    //char * workDir = "/storage/emulated/0/sdcard/ai_pic/";

    char * workDir = "/sdcard/ai_pic/";
    //char * workDir = "./";
	if(0 != access(workDir,0))
	{
	   __android_log_print(ANDROID_LOG_DEBUG, "access", "fail");
	   mkdir(workDir,777);
	}

     struct tm *t;
    time_t tt;
    time(&tt);
    t = localtime(&tt);
    int year =t->tm_year + 1900;
    int month = t->tm_mon + 1;
    int day = t->tm_mday ;
    int hour =t->tm_hour;
    int min = t->tm_min ;
    int sec =t->tm_sec ;
    static unsigned int nFileCount =0;
    char szTemp[128] ={0};
    sprintf (szTemp ,"%s%04d%02d%02d/" ,workDir, year ,month ,day );
    if(0 != access(szTemp,0))
	{
	    __android_log_print(ANDROID_LOG_DEBUG, "access", "fail");
        mkdir(szTemp,777);
	}
 __android_log_print(ANDROID_LOG_DEBUG, "snap", "szTemp:%s ", szTemp);

	char szSavePic[256] ={0};
	sprintf (szSavePic,"%s%02d_%02d_%02d_%d.jpg" , szTemp , hour ,min,sec ,nFileCount );

	//
    cv::Mat bgrimg ;
    cv::cvtColor(rgb ,bgrimg ,cv::COLOR_BGR2RGB);


    //cv::imwrite( (const char *)szSavePic ,bgrimg) ;
   // writetobmpfile(szSavePic ,bgrimg.data, rgb.cols ,rgb.rows ) ;

    nFileCount++;

   __android_log_print(ANDROID_LOG_DEBUG, "snap", "szSavePic:%s w:%d h:%d step:%d ", szSavePic ,
   rgb.cols  ,rgb.rows ,rgb.step[0]);

    return 1 ;
}



static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb ,double usetimes)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "time=%.2f ms [%d %d ]", usetimes,rgb.cols ,rgb.rows);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));



    return 0;
}

static SCRFD* g_scrfd = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;


};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // scrfd
     double usetimes =0 ;
    {
        ncnn::MutexLockGuard g(lock);
        if(SnapFlag==1)
        {
            SnapFlag =0;
            std::vector<FaceObject> faceobjects;
            g_scrfd->detect_rgba(rgb, faceobjects, szSaveRoiPic);
            return;
            write_img(rgb);
        }

        if (g_scrfd)
        {
            std::vector<FaceObject> faceobjects;
            double t1 =cv::getTickCount() ;
            g_scrfd->detect(rgb, faceobjects);
            //g_scrfd->detect_rgba(rgb, faceobjects, szSaveRoiPic);
            double t2=cv::getTickCount() ;

           usetimes  = (t2 -t1 ) /  cv::getTickFrequency() * 1000.0;
         if(g_modelid== 3)
         {
             if( faceobjects.size ()> 0 )
             {
                   __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "plate:%s w:%d h:%d step:%d", faceobjects[0].szinfo,rgb.cols ,rgb.rows ,rgb.step[0] );

                   //dt_raw_text(rgb.data ,rgb.cols ,rgb.rows, 64,128 ,faceobjects[0].szinfo ,strlen (faceobjects[0].szinfo) );
              }
         }
            g_scrfd->draw(rgb, faceobjects );
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb, usetimes);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_scrfd;
        g_scrfd = 0;
    }

    delete g_camera;
    g_camera = 0;
}

static int nModelLoad = 0;
// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }
    g_modelid = modelid ;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
    const char* modeltypes[] =
    {
        "face_kps",
        "moto",
        "plate",
        "head",
        "car",
        "biao",
    };

    const char* modeltype = modeltypes[(int)modelid];

    bool use_gpu = (int)cpugpu == 1;


    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_scrfd;
            g_scrfd = 0;
        }
        else
        {
            if (!g_scrfd)
             {
                g_scrfd = new SCRFD;
             }
            g_scrfd->load(mgr, modeltype, use_gpu);
        }
    }
    nModelLoad =1 ;
    return JNI_TRUE;
}

int nDetectBar = 0;
//JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_RecImg(JNIEnv* env, jobject thiz ,  jbyteArray pImage , jint width , jint height )
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_RecImg(JNIEnv* env, jobject thiz ,  jbyteArray pImage , jint width , jint height )
{
   signed char  *cimgbuf = env->GetByteArrayElements(pImage, JNI_FALSE);
   if(cimgbuf == NULL) return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "cimgbuf", "success");
   if(nModelLoad ==0)
   {
        env->ReleaseByteArrayElements( pImage ,cimgbuf ,0);
        return JNI_FALSE;
   }
    __android_log_print(ANDROID_LOG_DEBUG, "nModelLoad", "success");
    std::vector<FaceObject> faceobjects;
    double t1 =cv::getTickCount() ;
    cv::Mat rgb (  height ,width ,CV_8UC4 ,cimgbuf );
    nDetectBar = g_scrfd->detect_rgba(rgb, faceobjects, szSaveRoiPic);
    __android_log_print(ANDROID_LOG_DEBUG, "detect", "success");

    double t2=cv::getTickCount() ;

    env->ReleaseByteArrayElements( pImage ,cimgbuf ,0);
    __android_log_print(ANDROID_LOG_DEBUG, "release", "success");
    if(nDetectBar >0)  return JNI_TRUE;
    else return JNI_FALSE  ;

}

JNIEXPORT jstring  JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getSnapRoiImg(JNIEnv* env, jobject thiz)
{
    jstring jstr;

    if(nDetectBar >= 0)
    {
        jstr=env->NewStringUTF(szSaveRoiPic);
        //jstr=env->NewStringUTF( "detect");
    }
    else
    {
    jstr=env->NewStringUTF( "no detect");
    }

    return jstr ;
}

   // public native boolean  RecImg(byte[] pImg ,int width ,int height);
 //   public native  String  get_snap_roi_img( );
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_HandSnap(JNIEnv* env, jobject thiz)
{
    SnapFlag =1 ;

   __android_log_print(ANDROID_LOG_DEBUG, "snap", "snap pic");


    return  JNI_TRUE;
}


// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
