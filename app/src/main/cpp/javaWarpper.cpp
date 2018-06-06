#include <jni.h>
#include <string>
#include "include/Pipeline.h"

#include<android/bitmap.h>

std::string jstring2str(JNIEnv* env, jstring jstr) {

    char*   rtn   =   NULL;
    jclass   clsstring   =   env->FindClass("java/lang/String");
    jstring   strencode   =   env->NewStringUTF("GB2312");
    jmethodID   mid   =   env->GetMethodID(clsstring,   "getBytes",   "(Ljava/lang/String;)[B");
    jbyteArray   barr=   (jbyteArray)env->CallObjectMethod(jstr,mid,strencode);
    jsize   alen   =   env->GetArrayLength(barr);
    jbyte*   ba   =   env->GetByteArrayElements(barr,JNI_FALSE);
    if(alen > 0) {
        rtn   =   (char*)malloc(alen+1);
        memcpy(rtn,ba,alen);
        rtn[alen]=0;
    }
    env->ReleaseByteArrayElements(barr,ba,0);
    std::string stemp(rtn);
    free(rtn);
    return   stemp;
}

static struct {
    jclass jClassPoint;
    jmethodID jMethodInit;
    jfieldID jFieldIDX;
    jfieldID jFieldIDY;
} gPointInfo;

static jobject createJavaPoint(JNIEnv *env, cv::Point point_) {
    return env -> NewObject(gPointInfo.jClassPoint, gPointInfo.jMethodInit, point_.x, point_.y);
}

static void initClassInfo(JNIEnv *env) {
    gPointInfo.jClassPoint = reinterpret_cast<jclass>(env -> NewGlobalRef(env -> FindClass("android/graphics/Point")));
    gPointInfo.jMethodInit = env -> GetMethodID(gPointInfo.jClassPoint, "<init>", "(II)V");
    gPointInfo.jFieldIDX = env -> GetFieldID(gPointInfo.jClassPoint, "x", "I");
    gPointInfo.jFieldIDY = env -> GetFieldID(gPointInfo.jClassPoint, "y", "I");
}

extern "C" {

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv *env = NULL;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        return JNI_FALSE;
    }
    initClassInfo(env);
    return JNI_VERSION_1_4;
}

JNIEXPORT jlong JNICALL
Java_com_ynnqo_pr_PlateRecognition_InitPlateRecognizer(
        JNIEnv *env, jobject obj,
        jstring detector_filename,
        jstring finemapping_prototxt,
        jstring finemapping_caffemodel,
        jstring segmentation_prototxt,
        jstring segmentation_caffemodel,
        jstring charRecognization_proto,
        jstring charRecognization_caffemodel,
        jstring SegmentationFree_prototxt,
        jstring SegmentationFree_caffemodel) {

    std::string detector_path = jstring2str(env, detector_filename);
    std::string finemapping_prototxt_path = jstring2str(env, finemapping_prototxt);
    std::string finemapping_caffemodel_path = jstring2str(env, finemapping_caffemodel);
    std::string segmentation_prototxt_path = jstring2str(env, segmentation_prototxt);
    std::string segmentation_caffemodel_path = jstring2str(env, segmentation_caffemodel);
    std::string charRecognization_proto_path = jstring2str(env, charRecognization_proto);
    std::string charRecognization_caffemodel_path = jstring2str(env, charRecognization_caffemodel);
    std::string SegmentationFree_prototxt_path = jstring2str(env, SegmentationFree_prototxt);
    std::string SegmentationFree_caffemodel_path = jstring2str(env, SegmentationFree_caffemodel);


    pr::PipelinePR *PR = new pr::PipelinePR(detector_path,
                                            finemapping_prototxt_path,
                                            finemapping_caffemodel_path,
                                            segmentation_prototxt_path,
                                            segmentation_caffemodel_path,
                                            charRecognization_proto_path,
                                            charRecognization_caffemodel_path,
                                            SegmentationFree_prototxt_path,
                                            SegmentationFree_caffemodel_path);

    return (jlong) PR;

}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_SimpleRecognization(
        JNIEnv *env,
        jobject obj,
        jlong matPtr,
        jlong object_pr) {

    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;

    cv::Mat &mRgb = *(cv::Mat *) matPtr;
    cv::Mat rgb;
    cv::cvtColor(mRgb,rgb,cv::COLOR_RGBA2BGR);
    //cv::imwrite("/sdcard/demo.jpg",rgb);

    std::vector<pr::PlateInfo> list_res= PR->RunPiplineAsImage(rgb);
    std::string concat_results;
    for(auto one:list_res) {
        if (one.confidence>0.7)
            concat_results+=one.getPlateName()+",";
    }
    concat_results = concat_results.substr(0,concat_results.size()-1);

    return env->NewStringUTF(concat_results.c_str());

}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_ReleasePlateRecognizer(
        JNIEnv *env,
        jobject obj,
        jlong object_re) {
//    std::string hello = "Hello from C++";
    pr::PipelinePR *PR = (pr::PipelinePR *) object_re;
    delete PR;
}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_Test(
        JNIEnv *env,
        jobject obj) {

        std::string hello = "Hello from C++";
        return env->NewStringUTF(hello.c_str());

}


JNIEXPORT void JNICALL
Java_com_ynnqo_pr_PlateRecognition_GetPoint(
        JNIEnv *env,
        jobject obj,
        jlong matPtr,
        jlong object_pr,
        jobjectArray outPoint_) {

    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;

    cv::Mat &mRgb = *(cv::Mat *) matPtr;
    cv::Mat rgb;
    cv::cvtColor(mRgb,rgb,cv::COLOR_RGBA2BGR);

    if (env -> GetArrayLength(outPoint_) != 4) {
        return;
    }

    std::vector<cv::Point> scanPoints= PR->GetPointFromPlateRough(rgb);


    std::pair<std::string,float> pr;
    if (scanPoints.size() == 4) {
        if (scanPoints[0].x != scanPoints[3].x) {
            pr = PR->GetPlateResult(rgb, scanPoints);
        }
    }

    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }

}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_GetPoint2(
        JNIEnv *env,
        jobject obj,
        jlong matPtr,
        jlong object_pr,
        jobjectArray outPoint_) {

    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;

    cv::Mat &mRgb = *(cv::Mat *) matPtr;
    cv::Mat rgb;
    cv::cvtColor(mRgb,rgb,cv::COLOR_RGBA2BGR);

    std::vector<cv::Point> scanPoints= PR->GetPointFromPlateRough(rgb);

    std::pair<std::string,float> pr;
    if (scanPoints.size() == 4) {
        if (scanPoints[0].x != scanPoints[2].x) {
            pr = PR->GetPlateResult(rgb, scanPoints);
        }else{
            pr.first = "";
            pr.second = 0.00;
        }
    }

    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }


    return env->NewStringUTF(pr.first.c_str());


}




JNIEXPORT jint JNICALL
Java_com_ynnqo_pr_PlateRecognition_processBitmap(JNIEnv *env,
                                                          jclass,
                                                          jobject bmpObj,
                                                          jlong object_pr){



    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;


    void* pixelscolor;

    AndroidBitmapInfo bmpInfo={0};
    if(AndroidBitmap_getInfo(env,bmpObj,&bmpInfo)<0) {
        return -1;
    }


//    int* dataFromBmp=NULL;
//    if(AndroidBitmap_lockPixels(env,bmpObj,(void**)&dataFromBmp)) {
//        return -1;
//    }else{
//        // 此时 dataFromBmp 就是图片的首地址!!!
//        AndroidBitmap_unlockPixels(env,bmpObj);
//    }
//    return *dataFromBmp;


    if (bmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        //LOGE("Bitmap format is not RGBA_8888 !");
        return -1;
    }

    if (AndroidBitmap_lockPixels(env, bmpObj, &pixelscolor) < 0) {
        //LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return -1;
    }

    cv::Mat test2(bmpInfo.height,bmpInfo.width,CV_8UC4,(char*)pixelscolor);//BGRA

    cv::Mat rgb;
    cv::cvtColor(test2,rgb,cv::COLOR_RGBA2BGR);
    std::vector<cv::Point> scanPoints= PR->GetPointFromPlateRough(rgb);



    AndroidBitmap_unlockPixels(env,bmpObj);

    return 0;

}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_processBitmap2(JNIEnv *env,
                                                          jclass,
                                                          jobject bmpObj,
                                                          jlong object_pr,
                                                           jobjectArray outPoint_){



    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;


    void* pixelscolor;

    AndroidBitmapInfo bmpInfo={0};
    if(AndroidBitmap_getInfo(env,bmpObj,&bmpInfo)<0) {
        return env->NewStringUTF("error");
    }

    if (bmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        //LOGE("Bitmap format is not RGBA_8888 !");
        return env->NewStringUTF("error");
    }

    if (AndroidBitmap_lockPixels(env, bmpObj, &pixelscolor) < 0) {
        //LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return env->NewStringUTF("error");
    }

    AndroidBitmap_unlockPixels(env,bmpObj);

    cv::Mat test2(bmpInfo.height,bmpInfo.width,CV_8UC4,(char*)pixelscolor);//BGRA

    cv::Mat rgb;
    cv::cvtColor(test2,rgb,cv::COLOR_RGBA2BGR);

    std::vector<cv::Point> scanPoints= PR->GetPointFromPlateRough(rgb);

    std::pair<std::string,float> pr;
    if (scanPoints.size() == 4) {
        if (scanPoints[0].x != scanPoints[2].x) {
            pr = PR->GetPlateResult(rgb, scanPoints);
        }else{
            pr.first = "";
            pr.second = 0.00;
        }
    }

    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }

    return env->NewStringUTF(pr.first.c_str());

}

JNIEXPORT jstring JNICALL
Java_com_ynnqo_pr_PlateRecognition_processBitmap3(JNIEnv *env,
                                                  jclass,
                                                  jobject bmpObj,
                                                  jlong object_pr,
                                                  jobjectArray outPoint_){



    pr::PipelinePR *PR = (pr::PipelinePR *) object_pr;


    void* pixelscolor;

    AndroidBitmapInfo bmpInfo={0};
    if(AndroidBitmap_getInfo(env,bmpObj,&bmpInfo)<0) {
        return env->NewStringUTF("error");
    }

    if (bmpInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        //LOGE("Bitmap format is not RGBA_8888 !");
        return env->NewStringUTF("error");
    }

    if (AndroidBitmap_lockPixels(env, bmpObj, &pixelscolor) < 0) {
        //LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return env->NewStringUTF("error");
    }

    AndroidBitmap_unlockPixels(env,bmpObj);

    cv::Mat test2(bmpInfo.height,bmpInfo.width,CV_8UC4,(char*)pixelscolor);//BGRA

    cv::Mat rgb;
    cv::cvtColor(test2,rgb,cv::COLOR_RGBA2BGR);

    std::pair<std::string,std::vector<cv::Point>> pair = PR->GetPointFromPlateRough2(rgb);

    std::vector<cv::Point> scanPoints = pair.second;

    if (scanPoints.size() == 4) {
        for (int i = 0; i < 4; ++i) {
            env -> SetObjectArrayElement(outPoint_, i, createJavaPoint(env, scanPoints[i]));
        }
    }



    return env->NewStringUTF(pair.first.c_str());

}



}



