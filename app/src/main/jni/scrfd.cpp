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

#include "scrfd.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


#include "cpu.h"

#include "platemodel_cpp.h"
#include "plateid.h"

#include "car_model.h"
#include "carid.h"
#include "moto_only_model.h"
#include "moto_only_id.h"

#include "ast_rawosd.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include<unistd.h>
static int nInitOcr=0;
static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    //float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

//     #pragma omp parallel sections
    {
//         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
//         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const FaceObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const FaceObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

// insightface/detection/scrfd/mmdet/core/anchor/anchor_generator.py gen_single_level_base_anchors()
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = 0;
    const float cy = 0;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& kps_blob, float prob_threshold, std::vector<FaceObject>& faceobjects)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold)
                {
                    // insightface/detection/scrfd/mmdet/models/dense_heads/scrfd_head.py _get_bboxes_single()
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;

                    // insightface/detection/scrfd/mmdet/core/bbox/transforms.py distance2bbox()
                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    if (!kps_blob.empty())
                    {
                        const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);

                        obj.landmark[0].x = cx + kps.channel(0)[index] * feat_stride;
                        obj.landmark[0].y = cy + kps.channel(1)[index] * feat_stride;
                        obj.landmark[1].x = cx + kps.channel(2)[index] * feat_stride;
                        obj.landmark[1].y = cy + kps.channel(3)[index] * feat_stride;
                        obj.landmark[2].x = cx + kps.channel(4)[index] * feat_stride;
                        obj.landmark[2].y = cy + kps.channel(5)[index] * feat_stride;
                        obj.landmark[3].x = cx + kps.channel(6)[index] * feat_stride;
                        obj.landmark[3].y = cy + kps.channel(7)[index] * feat_stride;
                        obj.landmark[4].x = cx + kps.channel(8)[index] * feat_stride;
                        obj.landmark[4].y = cy + kps.channel(9)[index] * feat_stride;
                    }

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}


int SCRFD::load(const char* modeltype, bool use_gpu)
{
    scrfd.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    scrfd.opt = ncnn::Option();

#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif

    scrfd.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];


    sprintf(parampath, "%s-opt2.param", modeltype);
    sprintf(modelpath, "%s-opt2.bin", modeltype);



    has_kps = strstr(modeltype, "_kps") != NULL;

    if(strstr(modeltype, "face")!=NULL )
    {
        nModelType=0;
         scrfd.load_param(parampath);
         scrfd.load_model(modelpath);

    }
     if(strstr(modeltype, "moto")!=NULL )
     {
         nModelType=1;
     }
    if(strstr(modeltype, "plate")!=NULL )
     {
         nModelType=2;
        scrfd.load_param(plate_op_param_bin);
         scrfd.load_model(plate_op_bin);
     }

    if(strstr(modeltype, "head")!=NULL )
    {
           nModelType=3;
          scrfd.load_param_bin("location.param");
          scrfd.load_model("location.model");


     }
     if(strstr(modeltype, "car")!=NULL)
     {
            scrfd.load_param(car_opt_param_bin);
            scrfd.load_model(car_opt_bin);
             nModelType=4;
     }
    __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "nModelType:%d ", nModelType);

    return 0;
}

int SCRFD::load(AAssetManager* mgr, const char* modeltype, bool use_gpu)
{
    scrfd.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    scrfd.opt = ncnn::Option();


#if NCNN_VULKAN
    scrfd.opt.use_vulkan_compute = use_gpu;
#endif
    scrfd.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256]={0};
    char modelpath[256] ={0};

    sprintf(parampath, "%s-opt2.param", modeltype);
    sprintf(modelpath, "%s-opt2.bin", modeltype);
    has_kps = strstr(modeltype, "_kps") != NULL;
    if(has_kps)
    {
        scrfd.load_param(mgr, parampath);
        scrfd.load_model(mgr, modelpath);
        nModelType = 0;
    }

    int have_plate = strstr(modeltype, "plate") != NULL;
    int have_car = strstr(modeltype, "car") != NULL;
    int  have_moto = strstr(modeltype, "moto") != NULL;
    int havehead = strstr (modeltype ,"head" )!= NULL ;
   int  have_biao = strstr (modeltype ,"biao" )!= NULL ;




    if(have_plate)
    {
       scrfd.load_param(plate_op_param_bin);
       scrfd.load_model(plate_op_bin);


        nModelType=2;
    }
    if(havehead)
    {
       int nRet =  scrfd.load_param_bin(mgr, "location.param");
        __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "H load_param:%d ", nRet);
        nRet =   scrfd.load_model(mgr, "location.model");
         __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "H load_model:%d ", nRet);
        nModelType = 3;

            if(nInitOcr ==0 )
            {

                  int nRet =  ocrfd.load_param_bin(mgr,"ocr.param");
                   __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "load ocr param:%d ", nRet);
                   nRet =   ocrfd.load_model(mgr,"ocr.model");
                  __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "load ocr model:%d ", nRet);
                  nInitOcr=1 ;
            }


    }

    if(have_biao)
    {
            int nRet =  scrfd.load_param(mgr, "biao_opt.param");
             __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "H load_param:%d ", nRet);
              nRet =   scrfd.load_model(mgr, "biao_opt.bin");
             __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "H load_model:%d ", nRet);
              nModelType = 5;

    }


    if(have_car)
    {
        scrfd.load_param(car_opt_param_bin);
        scrfd.load_model(car_opt_bin);
         nModelType=4;
    }
    if(have_moto)
    {
            scrfd.load_param(moto_only_opt_param_bin);
            scrfd.load_model(moto_only_opt_bin);
            nModelType=1 ;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "LOAD_MODEL", "H nModelType:%d ", nModelType);

    return 0;
}
int _cmp_rect_conf(const void* r1, const void* r2)
{
	return ((OBJECT_RECT*)r1)->rects.left  - ((OBJECT_RECT*)r2)->rects.left;
}



#define CUTOFF 8            /* testing shows that this is good value */

static void swap (int *a, int *b, int width)
{
    int tmp;

    if ( a != b )
        /* Do the swap one character at a time to avoid potential alignment
           problems. */
        while ( width-- ) {
            tmp = *a;
            *a++ = *b;
            *b++ = tmp;
        }
}

static void shortsort (int *lo, int *hi, int width, int (*comp)(const void*, const void*))
{
    int *p, *max;

    /* Note: in assertions below, i and j are alway inside original bound of
       array to sort. */

    while (hi > lo) {
        /* A[i] <= A[j] for i <= j, j > hi */
        max = lo;
        for (p = lo+width; p <= hi; p += width) {
            /* A[i] <= A[max] for lo <= i < p */
            if (comp(p, max) > 0) {
                max = p;
            }
            /* A[i] <= A[max] for lo <= i <= p */
        }

        /* A[i] <= A[max] for lo <= i <= hi */

        swap(max, hi, width);

        /* A[i] <= A[hi] for i <= hi, so A[i] <= A[j] for i <= j, j >= hi */

        hi -= width;

        /* A[i] <= A[j] for i <= j, j > hi, loop top condition established */
    }
    /* A[i] <= A[j] for i <= j, j > lo, which implies A[i] <= A[j] for i < j,
       so array is sorted */
}

void my_sort(void *base, int num, int width, int (*comp)(const void*, const void*))
{
    int *lo, *hi;              /* ends of sub-array currently sorting */
    int *mid;                  /* points to middle of subarray */
    int *loguy, *higuy;        /* traveling pointers for partition step */
    int size;              /* size of the sub-array */
    int *lostk[30], *histk[30];
    int stkptr;                 /* stack for saving sub-array to be processed */

    /* Note: the number of stack entries required is no more than
       1 + log2(size), so 30 is sufficient for any array */

    if (num < 2 || width == 0)
        return;                 /* nothing to do */

	width >>= 2;	// div 4

    stkptr = 0;                 /* initialize stack */

    lo = (int*)base;
    hi = (int*)base + width * (num-1);        /* initialize limits */

    /* this entry point is for pseudo-recursion calling: setting
       lo and hi and jumping to here is like recursion, but stkptr is
       prserved, locals aren't, so we preserve stuff on the stack */
recurse:

    size = (hi - lo) / width + 1;        /* number of el's to sort */

    /* below a certain size, it is faster to use a O(n^2) sorting method */
    if (size <= CUTOFF) {
         shortsort(lo, hi, width, comp);
    }
    else {
        /* First we pick a partititioning element.  The efficiency of the
           algorithm demands that we find one that is approximately the
           median of the values, but also that we select one fast.  Using
           the first one produces bad performace if the array is already
           sorted, so we use the middle one, which would require a very
           wierdly arranged array for worst case performance.  Testing shows
           that a median-of-three algorithm does not, in general, increase
           performance. */

        mid = lo + (size / 2) * width;      /* find middle element */
        swap(mid, lo, width);               /* swap it to beginning of array */

        /* We now wish to partition the array into three pieces, one
           consisiting of elements <= partition element, one of elements
           equal to the parition element, and one of element >= to it.  This
           is done below; comments indicate conditions established at every
           step. */

        loguy = lo;
        higuy = hi + width;

        /* Note that higuy decreases and loguy increases on every iteration,
           so loop must terminate. */
        for (;;) {
            /* lo <= loguy < hi, lo < higuy <= hi + 1,
               A[i] <= A[lo] for lo <= i <= loguy,
               A[i] >= A[lo] for higuy <= i <= hi */

            do  {
                loguy += width;
            } while (loguy <= hi && comp(loguy, lo) <= 0);

            /* lo < loguy <= hi+1, A[i] <= A[lo] for lo <= i < loguy,
               either loguy > hi or A[loguy] > A[lo] */

            do  {
                higuy -= width;
            } while (higuy > lo && comp(higuy, lo) >= 0);

            /* lo-1 <= higuy <= hi, A[i] >= A[lo] for higuy < i <= hi,
               either higuy <= lo or A[higuy] < A[lo] */

            if (higuy < loguy)
                break;

            /* if loguy > hi or higuy <= lo, then we would have exited, so
               A[loguy] > A[lo], A[higuy] < A[lo],
               loguy < hi, highy > lo */

            swap(loguy, higuy, width);

            /* A[loguy] < A[lo], A[higuy] > A[lo]; so condition at top
               of loop is re-established */
        }

        /*     A[i] >= A[lo] for higuy < i <= hi,
               A[i] <= A[lo] for lo <= i < loguy,
               higuy < loguy, lo <= higuy <= hi
           implying:
               A[i] >= A[lo] for loguy <= i <= hi,
               A[i] <= A[lo] for lo <= i <= higuy,
               A[i] = A[lo] for higuy < i < loguy */

        swap(lo, higuy, width);     /* put partition element in place */

        /* OK, now we have the following:
              A[i] >= A[higuy] for loguy <= i <= hi,
              A[i] <= A[higuy] for lo <= i < higuy
              A[i] = A[lo] for higuy <= i < loguy    */

        /* We've finished the partition, now we want to sort the subarrays
           [lo, higuy-1] and [loguy, hi].
           We do the smaller one first to minimize stack usage.
           We only sort arrays of length 2 or more.*/

        if ( higuy - 1 - lo >= hi - loguy ) {
            if (lo + width < higuy) {
                lostk[stkptr] = lo;
                histk[stkptr] = higuy - width;
                ++stkptr;
            }                           /* save big recursion for later */

            if (loguy < hi) {
                lo = loguy;
                goto recurse;           /* do small recursion */
            }
        }
        else {
            if (loguy < hi) {
                lostk[stkptr] = loguy;
                histk[stkptr] = hi;
                ++stkptr;               /* save big recursion for later */
            }

            if (lo + width < higuy) {
                hi = higuy - width;
                goto recurse;           /* do small recursion */
            }
        }
    }

    /* We have sorted the array, except for any pending sorts on the stack.
       Check if there are any, and do them. */

    --stkptr;
    if (stkptr >= 0) {
        lo = lostk[stkptr];
        hi = histk[stkptr];
        goto recurse;           /* pop subarray from stack */
    }
    else
        return;                 /* all subarrays done */
}

char szchars[][4] = {
"0","1","2","3","4","5","6","7","8","9",
"A","B","C","D","E","F","G","H","J",
"K","L","M","N","P","Q","R","S","T",
"U","V","W","X","Y","Z",
};


 int write_img_to_file (cv::Mat &rgb ,char *szFiles, int nlabel, int x, int y, int w, int h)
{
    //char * workDir = "/storage/emulated/0/sdcard/ai_pic/";

    char * workDir = "/sdcard/ai_pic/";
    //char * workDir = "";
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
    sprintf (szTemp ,"%s" ,workDir);
    //
    //sprintf (szTemp ,"%s%04d%02d%02d/" ,workDir, year ,month ,day );
    if(0 != access(szTemp,0))
	{
	    __android_log_print(ANDROID_LOG_DEBUG, "access", "fail");
        mkdir(szTemp,777);
	}
 __android_log_print(ANDROID_LOG_DEBUG, "snap", "szTemp:%s ", szTemp);

	char szSavePic[256] ={0};
	char szSavePic2[256] ={0};
	//sprintf (szSavePic,"%s%d_%04d_%04d_%04d_%04d_%d.jpg" , szTemp ,nFileCount, x ,y,w ,h, nlabel);
	sprintf (szSavePic2,"%s%d_%d_%d_%d_%d.jpg" ,szTemp, x ,y,w ,h, nlabel);
	sprintf (szSavePic,"%d_%d_%d_%d_%d" ,x ,y,w ,h, nlabel);

	//w
    cv::Mat bgrimg ;
    cv::cvtColor(rgb ,bgrimg ,cv::COLOR_RGBA2RGB);

    cv::imwrite( (const char *)szSavePic2 ,bgrimg) ;
    //writetobmpfile(szSavePic ,bgrimg.data, rgb.cols ,rgb.rows ) ;

    nFileCount++;

   __android_log_print(ANDROID_LOG_DEBUG, "ROI", "szSavePic:%s w:%d h:%d step:%d ", szSavePic ,
   rgb.cols  ,rgb.rows ,rgb.step[0]);

   //strcpy( szFiles , szSavePic);
   strcat(szFiles, szSavePic);
   strcat(szFiles, " ");

    return 1 ;
}



int SCRFD::detect_rgba(  const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, char * szSaveRoiPic ,float prob_threshold , float nms_threshold )
{
    int width =  rgb.cols;
    int height = rgb.rows;
    int w = width ;
    int h = height ;

    *szSaveRoiPic = {0};
    int  target_size= 352 ;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
    const float mean_vals[3] = { 0,0,0 };
    const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = scrfd.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
    int nObCount = out.h;
    int nHaveBar = 0;
     int order = 0;
     int x[50] = {0};
     int y[50] = {0};
     for (int i = 0; i < out.h; i++)
      {
             const float* values = out.row(i);
             if (values[1] < 0.65 ) continue;
             FaceObject  obj ;
             obj.rect.x = values[2] * w;
             obj.rect.y = values[3] * h;
             obj.rect.width  =  values[4] * w - obj.rect.x ;
             obj.rect.height  = values[5] * h - obj.rect.y ;
             obj.prob =  values[1];
             obj.nlabel = (int)values[0];
             if(obj.nlabel == 2)
             {
                 order++;
                 cv::Rect roi  ;
                 roi.x = obj.rect.x  ;
                 roi.y = obj.rect.y ;
                 roi.width =  obj.rect.width ;
                 roi.height  = obj.rect.height ;
                 int x_center = obj.rect.x + obj.rect.width / 2;
                 int y_center = obj.rect.y + obj.rect.height / 2;

                 x[i] = x_center;
                 y[i] = y_center;
                 roi.width = obj.rect.width * 1.5;
                 roi.height = obj.rect.height * 1.5;
                 roi.x = x_center - roi.width / 2 > 0 ? x_center - roi.width / 2 : 0;
                 roi.y = y_center - roi.height / 2 > 0 ? y_center - roi.height / 2 : 0;
                 __android_log_print(ANDROID_LOG_DEBUG, "rgb", "w:%d h:%d ", rgb.cols ,
                                  rgb.rows);
                 __android_log_print(ANDROID_LOG_DEBUG, "Point", "x:%d y:%d w:%d h:%d ", roi.x ,
                 roi.y ,roi.width,roi.height);

                 cv::Mat roiimg =rgb (roi ).clone() ;
                 write_img_to_file (
                 roiimg,szSaveRoiPic, order, roi.x, roi.y,roi.width,roi.height);
                  nHaveBar=1 ;
             }

             faceobjects.push_back(obj) ;
    }
    x[50] = {0};
    y[50] = {0};
    __android_log_print(ANDROID_LOG_DEBUG, "Path", "%s", szSaveRoiPic);
    return nHaveBar;
 }

int SCRFD::detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold)
{
    int width =  rgb.cols;
    int height = rgb.rows;
    int w = width ;
    int h = height ;
    // insightface/detection/scrfd/configs/scrfd/scrfd_500m.py
    int target_size = 640;
    if(nModelType == 2 )
     {
        target_size= 416 ;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
        const float mean_vals[3] = { 0,0,0 };
        const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = scrfd.create_extractor();
        ex.set_light_mode(true);

        ex.input(plate_op_param_id::LAYER_data, in);
         ncnn::Mat out;
         ex.extract( plate_op_param_id::BLOB_output, out);

         int nObCount = out.h;

         for (int i = 0; i < out.h; i++)
         {
         		const float* values = out.row(i);
         		if (values[1] < 0.35 ) continue;
                FaceObject  obj ;
                obj.rect.x = values[2] * w;
                obj.rect.y = values[3] * h;
                obj.rect.width  =  values[4] * w - obj.rect.x ;
                obj.rect.height  = values[5] * h - obj.rect.y ;
                obj.prob = values[1];
                faceobjects.push_back(obj) ;
         }

      //   __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "PlateCount:%d ", faceobjects.size());


     }
      if(nModelType ==1)
      {
        target_size= 416 ;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
        const float mean_vals[3] = { 0,0,0 };
        const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = scrfd.create_extractor();
        ex.set_light_mode(true);

        ex.input(moto_only_opt_param_id::LAYER_data, in);
         ncnn::Mat out;
         ex.extract(moto_only_opt_param_id::BLOB_output, out);

         int nObCount = out.h;

         for (int i = 0; i < out.h; i++)
         {
         		const float* values = out.row(i);
         		if (values[1] < 0.35 ) continue;
                FaceObject  obj ;
                obj.rect.x = values[2] * w;
                obj.rect.y = values[3] * h;
                obj.rect.width  =  values[4] * w - obj.rect.x ;
                obj.rect.height  = values[5] * h - obj.rect.y ;
                obj.prob = values[1];
                faceobjects.push_back(obj) ;
         }

       //  __android_log_print(ANDROID_LOG_DEBUG, "MOTOS", "MOTO:%d ", faceobjects.size());

      }


       if(nModelType ==3)
       {

                    target_size= 320 ;
                    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
                    const float mean_vals[3] = { 0,0,0 };
                    const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
                    in.substract_mean_normalize(mean_vals, norm_vals);
                    ncnn::Extractor ex = scrfd.create_extractor();
                    ex.set_light_mode(true);

                    ex.input(0, in);
                     ncnn::Mat out;
                     ex.extract( 61, out);

                     int nObCount = out.h;
                     for (int i = 0; i < out.h; i++)
                     {
                            const float* values = out.row(i);
                            if (values[1] < 0.35 ) continue;
                            FaceObject  obj ;
                            obj.rect.x = values[2] * w;
                            obj.rect.y = values[3] * h;
                            obj.rect.width  =  values[4] * w - obj.rect.x ;
                            obj.rect.height  = values[5] * h - obj.rect.y ;
                            obj.prob = values[1];

                            if(obj.rect.x <0) obj.rect.x =0;
                             if(obj.rect.y <0) obj.rect.y =0;
                             if(obj.rect.width +obj.rect.x > width-1 ) obj.rect.width = width -obj.rect.x -1 ;
                            if(obj.rect.height  +obj.rect.y > height -1 ) obj.rect.width = height -obj.rect.y -1 ;

                            cv::Mat roi = rgb(obj.rect ).clone() ;

                            int  target_size = 224;
                            int width = roi.cols ;
                            int height = roi.rows ;

                            ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data , ncnn::Mat::PIXEL_BGR2RGB,roi.cols, roi.rows , target_size, target_size);


                             float mean_vals[3] = { 0,0,0 };
                             float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };

                            in.substract_mean_normalize(mean_vals, norm_vals);

                            ncnn::Extractor ex = ocrfd.create_extractor();
                            ex.set_light_mode(true);

                            ex.input(0, in);
                            ncnn::Mat out;
                            	//ex.extract(plate_op_param_id::BLOB_output, out);
                            ex.extract(61, out);
                            OBJECT_RECT  pObject[32] ={0} ;
                            int nObCount = out.h;
                            int t = 0;

                            for (int i = 0; i < out.h; i++)
                            {
                            		const float* values = out.row(i);
                            		if (values[1] < 0.45) continue;

                            		pObject[t].conf = values[1];
                            		pObject[t].label = values[0];

                            		pObject[t].rects.left = values[2] * width;
                            		pObject[t].rects.top = values[3] * height ;

                            		pObject[t].rects.right = values[4] * width;
                            		pObject[t].rects.bottom = values[5] * height;

                            		int  w = pObject[t].rects.right - pObject[t].rects.left;
                            		int  h = pObject[t].rects.bottom - pObject[t].rects.top;

                            		if (pObject[t].rects.left < 0)pObject[t].rects.left = 0;
                            		if (pObject[t].rects.top < 0)pObject[t].rects.top = 0;
                            		if (pObject[t].rects.right > width - 2)pObject[t].rects.right = width - 2;
                            		if (pObject[t].rects.bottom > height - 2)pObject[t].rects.bottom = height - 2;
                            		t++;
                            }

		                     my_sort(pObject, t, sizeof(OBJECT_RECT), _cmp_rect_conf);
                             memset(obj.szinfo,0,32) ;

                             if(t>=7)
                             {
                                    for(int x =0;x<t;x++)
                                     {
                                           strcat(obj.szinfo , szchars[pObject[x].label-1]);
                                     }
                             }
                               //  __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "plate:%s ", obj.szinfo);
                              faceobjects.push_back(obj) ;


                     }
                     if(faceobjects.size() >0 )
                     {

                     }
       }
 if(nModelType ==4)
 {
        target_size= 320 ;
       ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
        const float mean_vals[3] = { 0,0,0 };
        const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = scrfd.create_extractor();
        ex.set_light_mode(true);


        ex.input(car_opt_param_id::LAYER_data, in);
        ncnn::Mat out;
        ex.extract(car_opt_param_id::BLOB_output, out);
         int nObCount = out.h;

         for (int i = 0; i < out.h; i++)
         {
         		const float* values = out.row(i);
         		if (values[1] < 0.35 ) continue;
                FaceObject  obj ;
                obj.rect.x = values[2] * w;
                obj.rect.y = values[3] * h;
                obj.rect.width  =  values[4] * w - obj.rect.x ;
                obj.rect.height  = values[5] * h - obj.rect.y ;
                obj.prob = values[1];
                faceobjects.push_back(obj) ;
         }

     //     __android_log_print(ANDROID_LOG_DEBUG, "PLATE", "CarCout:%d ", faceobjects.size());
//

 }

 if(nModelType == 5)
 {
     target_size= 352 ;
       ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB,  width, height, target_size, target_size);
        const float mean_vals[3] = { 0,0,0 };
        const float norm_vals[3] = { 1 / 256.0, 1 / 256.0, 1 / 256.0 };
        in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = scrfd.create_extractor();
        ex.set_light_mode(true);

        ex.input("data", in);
        ncnn::Mat out;
        ex.extract("output", out);
         int nObCount = out.h;

         for (int i = 0; i < out.h; i++)
         {
         		const float* values = out.row(i);
         		if (values[1] < 0.45 ) continue;
                FaceObject  obj ;
                obj.rect.x = values[2] * w;
                obj.rect.y = values[3] * h;
                obj.rect.width  =  values[4] * w - obj.rect.x ;
                obj.rect.height  = values[5] * h - obj.rect.y ;
                obj.prob =  values[1];
                obj.nlabel = (int)values[0];
                __android_log_print(ANDROID_LOG_DEBUG, "BIAO", "label:%d ",obj.nlabel);
                int x_center = (obj.rect.x + obj.rect.width) / 2;
                int y_center = (obj.rect.y + obj.rect.height) / 2;
                obj.rect.width = obj.rect.width * 1.2;
                obj.rect.height = obj.rect.height * 1.2;
                faceobjects.push_back(obj) ;
         }

 }
    if(nModelType ==0)
    {
     // pad to multiple of 32
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = h * scale;
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

        // pad to target_size rectangle
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = scrfd.create_extractor();

        ex.input("input.1", in_pad);

        std::vector<FaceObject> faceproposals;

        // stride 8
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_8", score_blob);
            ex.extract("bbox_8", bbox_blob);
            if (has_kps)
                ex.extract("kps_8", kps_blob);

            const int base_size = 16;
            const int feat_stride = 8;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceObject> faceobjects32;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects32);

            faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
        }

        // stride 16
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_16", score_blob);
            ex.extract("bbox_16", bbox_blob);
            if (has_kps)
                ex.extract("kps_16", kps_blob);

            const int base_size = 64;
            const int feat_stride = 16;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceObject> faceobjects16;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects16);

            faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
        }

        // stride 32
        {
            ncnn::Mat score_blob, bbox_blob, kps_blob;
            ex.extract("score_32", score_blob);
            ex.extract("bbox_32", bbox_blob);
            if (has_kps)
                ex.extract("kps_32", kps_blob);

            const int base_size = 256;
            const int feat_stride = 32;
            ncnn::Mat ratios(1);
            ratios[0] = 1.f;
            ncnn::Mat scales(2);
            scales[0] = 1.f;
            scales[1] = 2.f;
            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

            std::vector<FaceObject> faceobjects8;
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects8);

            faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(faceproposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(faceproposals, picked, nms_threshold);

        int face_count = picked.size();

        faceobjects.resize(face_count);
        for (int i = 0; i < face_count; i++)
        {
            faceobjects[i] = faceproposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (faceobjects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (faceobjects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - (hpad / 2)) / scale;

            x0 = std::max(std::min(x0, (float)width - 1), 0.f);
            y0 = std::max(std::min(y0, (float)height - 1), 0.f);
            x1 = std::max(std::min(x1, (float)width - 1), 0.f);
            y1 = std::max(std::min(y1, (float)height - 1), 0.f);

            faceobjects[i].rect.x = x0;
            faceobjects[i].rect.y = y0;
            faceobjects[i].rect.width = x1 - x0;
            faceobjects[i].rect.height = y1 - y0;

            if (has_kps)
            {
                float x0 = (faceobjects[i].landmark[0].x - (wpad / 2)) / scale;
                float y0 = (faceobjects[i].landmark[0].y - (hpad / 2)) / scale;
                float x1 = (faceobjects[i].landmark[1].x - (wpad / 2)) / scale;
                float y1 = (faceobjects[i].landmark[1].y - (hpad / 2)) / scale;
                float x2 = (faceobjects[i].landmark[2].x - (wpad / 2)) / scale;
                float y2 = (faceobjects[i].landmark[2].y - (hpad / 2)) / scale;
                float x3 = (faceobjects[i].landmark[3].x - (wpad / 2)) / scale;
                float y3 = (faceobjects[i].landmark[3].y - (hpad / 2)) / scale;
                float x4 = (faceobjects[i].landmark[4].x - (wpad / 2)) / scale;
                float y4 = (faceobjects[i].landmark[4].y - (hpad / 2)) / scale;

                faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
                faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
                faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
                faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
                faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
                faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
                faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
                faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
                faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
                faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
            }
        }

    }


    return 0;
}

int SCRFD::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

//         fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        if(obj.nlabel ==1 )
              cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0),2 );
         else
             cv::rectangle(rgb, obj.rect, cv::Scalar(0, 0, 255),2 );


        if (has_kps)
        {
            cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%",  obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        if(nModelType==3)
         {

            dt_raw_text((unsigned char *)rgb.data ,rgb.cols , rgb.rows, x  ,y+ obj.rect.height , (char*)obj.szinfo ,strlen(obj.szinfo), 4 );
         }


    }

    return 0;
}
