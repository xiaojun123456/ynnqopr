package com.ynnqo.pr;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.util.Log;

/**
 * Created by yujinke on 24/10/2017.
 */

public class PlateRecognition {

    static {
        System.loadLibrary("hyperlpr");
    }

    static native long InitPlateRecognizer(String casacde_detection,
                                           String finemapping_prototxt,
                                           String finemapping_caffemodel,
                                           String segmentation_prototxt,
                                           String segmentation_caffemodel,
                                           String charRecognization_proto,
                                           String charRecognization_caffemodel,
                                           String SegmentationFree_prototxt,
                                           String SegmentationFree_caffemodel);

    static native long ReleasePlateRecognizer(long  object);

    static native String SimpleRecognization(long inputMat,long object);

    static native String Test();

    public static Point[] scan(long inputMat,long object) {
        Point[] outPoints = new Point[4];

        //GetPoint(inputMat,object, outPoints);

        String str = GetPoint2(inputMat,object, outPoints);
        Log.e("str--","-----"+str+"---------");

        return outPoints;
    }

    public static Point[] scan2(Bitmap bitmap,long object) {
        Point[] outPoints = new Point[4];


        try{

            //端到端识别
            //String str = processBitmap2(bitmap,object, outPoints);

            //字符识别识别
            String str = processBitmap3(bitmap,object, outPoints);

            Log.e("str--","-----"+str+"---------");

        }catch (Exception e){
            Log.e("str--","-----error---------");
        }



        return outPoints;
    }

    static native void GetPoint(long inputMat,long object, Point[] outPoints);
    static native String GetPoint2(long inputMat,long object, Point[] outPoints);


    static native int processBitmap(Bitmap bitmap,long object);

    static native String processBitmap2(Bitmap bitmap,long object, Point[] outPoints);

    static native String processBitmap3(Bitmap bitmap,long object, Point[] outPoints);

}
