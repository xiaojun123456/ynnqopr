package com.ynnqo.pr;


import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.media.ExifInterface;
import android.os.Environment;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class CameraView
        extends SurfaceView
        implements SurfaceHolder.Callback, Camera.PreviewCallback {

    private final String TAG = "CameraView";
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private boolean isPreviewOn;

    //默认预览尺寸
    private int imageWidth = 1920;
    private int imageHeight = 1080;

    //帧率
    private int frameRate = 30;

    private boolean isScanning = false;

    private Boolean isDuijiao = false;

    private Context cxt;

    private int dg;



    public long handle;

    public void copyFilesFromAssets(Context context, String oldPath, String newPath) {

        try {

            String[] fileNames = context.getAssets().list(oldPath);

            if (fileNames.length > 0) {

                // directory
                File file = new File(newPath);
                if (!file.mkdir()) {
                    Log.d("mkdir","can't make folder");
                }

                for (String fileName : fileNames) {
                    copyFilesFromAssets(context, oldPath + "/" + fileName, newPath + "/" + fileName);
                }

            } else {

                // file
                InputStream is = context.getAssets().open(oldPath);
                FileOutputStream fos = new FileOutputStream(new File(newPath));
                byte[] buffer = new byte[1024];
                int byteCount;
                while ((byteCount = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, byteCount);
                }
                fos.flush();
                is.close();
                fos.close();

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void initRecognizer(Context cxt) {

        String assetPath = "pr";
        String sdcardPath = Environment.getExternalStorageDirectory() + File.separator + assetPath;

        copyFilesFromAssets(cxt, assetPath, sdcardPath);

        String cascade_filename  =  sdcardPath + File.separator+"cascade.xml";
        String finemapping_prototxt  =  sdcardPath + File.separator+"HorizonalFinemapping.prototxt";
        String finemapping_caffemodel  =  sdcardPath + File.separator+"HorizonalFinemapping.caffemodel";
        String segmentation_prototxt =  sdcardPath + File.separator+"Segmentation.prototxt";
        String segmentation_caffemodel =  sdcardPath + File.separator+"Segmentation.caffemodel";
        String character_prototxt =  sdcardPath + File.separator+"CharacterRecognization.prototxt";
        String character_caffemodel=  sdcardPath + File.separator+"CharacterRecognization.caffemodel";

        String SegmentationFree_prototxt =  sdcardPath + File.separator+"SegmentationFree.prototxt";
        String SegmentationFree_caffemodel=  sdcardPath + File.separator+"SegmentationFree.caffemodel";

        handle  =  PlateRecognition.InitPlateRecognizer(
                cascade_filename,
                finemapping_prototxt,finemapping_caffemodel,
                segmentation_prototxt,segmentation_caffemodel,
                character_prototxt,character_caffemodel,
                SegmentationFree_prototxt,SegmentationFree_caffemodel
        );

        System.out.println("333333333333333333333");

    }


    public CameraView(Context context) {
        super(context);
        Log.e("---","11111");
        init();
    }

    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        Log.e("---","22222");
        init();
        cxt = context;
        //initRecognizer(context);
    }

    public CameraView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        Log.e("---","33333");
        init();
    }


    public void setDG(int dg){
        this.dg =dg;
        if (mCamera != null)
            initCameraParams();

    }

    private void init() {
        Log.e("---","4444");
        mHolder = getHolder();
        //设置SurfaceView 的SurfaceHolder的回调函数
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }




    @Override
    public void surfaceCreated(SurfaceHolder holder) {

        Log.e("---","5555");

        //Surface创建时开启Camera
        openCamera();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

        Log.e("---","66666");

        //设置Camera基本参数
        if (mCamera != null)
            initCameraParams();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {

        Log.e("---","77777");

        try {
            release();
        } catch (Exception e) {
        }

        closeCamera();

    }


    int k = 0;

    /**
     * Camera帧数据回调用
     */
    @Override
    public void onPreviewFrame(final byte[] data, final Camera camera) {

        //Log.e("------------","onPreviewFrame");

        if(isDuijiao){
            isDuijiao = false;
            mCamera.autoFocus(autoFocusCB);

            Log.e("true","--------------------");


        if (!isScanning) {
            isScanning = true;
            new Thread(new Runnable(){
                @Override
                public void run() {

                    try {

                        Log.e("true","111111");

                        //获取Camera预览尺寸
                        Camera.Size size = camera.getParameters().getPreviewSize();

                        //将帧数据转为bitmap
                        YuvImage image = new YuvImage(data, ImageFormat.NV21, size.width, size.height, null);

                        if (image != null) {

                            Log.e("true","2222222");

                            ByteArrayOutputStream stream = new ByteArrayOutputStream();

                            //将帧数据转为图片（new Rect()是定义一个矩形提取区域，我这里是提取了整张图片，
                            // 然后旋转90度后再才裁切出需要的区域，效率会较慢，实际使用的时候，
                            // 照片默认横向的,可以直接计算逆向90°时，left、top的值，然后直接提取需要区域，
                            // 提出来之后再压缩、旋转 速度会快一些）
                            image.compressToJpeg(new Rect(0, 0, size.width, size.height), 95, stream);

                            Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());

                            stream.close();




                            Camera.CameraInfo info = new Camera.CameraInfo();
                            Camera.getCameraInfo(0, info);
                            int result;
                            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                                result = (info.orientation + dg) % 360;
                                result = (360 - result) % 360;  // compensate the mirror
                            } else {  // back-facing
                                result = (info.orientation - dg + 360) % 360;
                            }

                            //这里返回的照片默认横向的，先将图片旋转90度
                            bmp = rotateToDegrees(bmp, result);

                            //然后裁切出需要的区域，具体区域要和UI布局中配合，这里取图片正中间，宽度取图片的一半，高度这里用的适配数据，可以自定义
                            //bmp = bitmapCrop(bmp, bmp.getWidth() / 4, bmp.getHeight() / 2 - (int) getResources().getDimension(R.dimen.x25), bmp.getWidth() / 2, (int) getResources().getDimension(R.dimen.x50));
                            //bmp = bitmapCrop(bmp, bmp.getWidth() / 4, bmp.getHeight() / 2 - 50, bmp.getWidth() / 2, 50);

                            if (bmp == null){return;}

                            String PATH = Environment.getExternalStorageDirectory().toString() + "/xxx/";
                            File file = new File(PATH);
                            if (!file.exists())
                                file.mkdirs();

                            saveImage(bmp,PATH+k+".jpg");

                            Intent it = new Intent("dd");
                            it.putExtra("path",PATH+k+".jpg");
                            cxt.sendBroadcast(it);


                            //Mat mat_src = new Mat(bmp.getWidth(), bmp.getHeight(), CvType.CV_8UC4);
                            //float new_w = bmp.getWidth();
                            //float new_h = bmp.getHeight();
                            //Size sz = new Size(new_w,new_h);
                            //Utils.bitmapToMat(bmp, mat_src);
                            //Imgproc.resize(mat_src,mat_src,sz);

                            //String res = PlateRecognition.SimpleRecognization(mat_src.getNativeObjAddr(),handle);
                            //Log.e("rv",res+"---------");

                            //Point[] pts = PlateRecognition.scan(mat_src.getNativeObjAddr(),handle);
                            //Log.e("rv",pts[0].x+"---------"+pts[0].y);
                            //setCropPoints(pts);



                            Log.e("true","333333");

                            //将裁切的图片显示出来（测试用，需要为CameraView  setTag（ImageView））
                            //ImageView imageView = (ImageView) getTag();
                            //imageView.setImageBitmap(bmp);


                            k++;

                            isScanning = false;


                            //开始识别
//                            OcrUtil.ScanEnglish(bmp, new MyCallBack() {
//                                @Override
//                                public void response(String result) {
//                                    //这是区域内扫除的所有内容
//                                    Log.d("scantest", "扫描结果：  " + result);
//                                    //检索结果中是否包含手机号
//                                    Log.d("scantest", "手机号码：  " + getTelnum(result));
//
//                                    isScanning = false;
//                                }
//                            });

                        }

                        //Thread.sleep(500);

                    } catch (Exception ex) {
                        isScanning = false;
                    }



                }
            }).start();

        }










        }else{

            //Log.e("false","--------------------");

        }


    }

    public void onPreviewFrame2(final byte[] data, final Camera camera) {

        //Log.e("------------","onPreviewFrame");

        if (!isScanning) {
            isScanning = true;
            new Thread(new Runnable(){
                @Override
                public void run() {

                    try {

                        Log.e("true","111111");

                        //获取Camera预览尺寸
                        Camera.Size size = camera.getParameters().getPreviewSize();

                        //将帧数据转为bitmap
                        YuvImage image = new YuvImage(data, ImageFormat.NV21, size.width, size.height, null);

                        if (image != null) {

                            Log.e("true","2222222");

                            ByteArrayOutputStream stream = new ByteArrayOutputStream();

                            //将帧数据转为图片（new Rect()是定义一个矩形提取区域，我这里是提取了整张图片，
                            // 然后旋转90度后再才裁切出需要的区域，效率会较慢，实际使用的时候，
                            // 照片默认横向的,可以直接计算逆向90°时，left、top的值，然后直接提取需要区域，
                            // 提出来之后再压缩、旋转 速度会快一些）
                            image.compressToJpeg(new Rect(0, 0, size.width, size.height), 80, stream);

                            Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());

                            stream.close();



                            Camera.CameraInfo info = new Camera.CameraInfo();
                            Camera.getCameraInfo(0, info);
                            int result;
                            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                                result = (info.orientation + dg) % 360;
                                result = (360 - result) % 360;  // compensate the mirror
                            } else {  // back-facing
                                result = (info.orientation - dg + 360) % 360;
                            }
                            //这里返回的照片默认横向的，先将图片旋转90度
                            bmp = rotateToDegrees(bmp, result);

                            if (bmp == null){return;}

                            String PATH = Environment.getExternalStorageDirectory().toString() + "/xxx/";
                            File file = new File(PATH);
                            if (!file.exists())
                                file.mkdirs();

                            saveImage(bmp,PATH+k+".jpg");

                            Intent it = new Intent("dd");
                            it.putExtra("path",PATH+k+".jpg");
                            cxt.sendBroadcast(it);

                            k++;

                            isScanning = false;

                        }

                    } catch (Exception ex) {
                        isScanning = false;
                    }



                }
            }).start();

        }




    }



    /**
     * Bitmap裁剪
     *
     * @param bitmap 原图
     * @param width  宽
     * @param height 高
     */
    public static Bitmap bitmapCrop(Bitmap bitmap, int left, int top, int width, int height) {
        if (null == bitmap || width <= 0 || height < 0) {
            return null;
        }
        int widthOrg = bitmap.getWidth();
        int heightOrg = bitmap.getHeight();
        if (widthOrg >= width && heightOrg >= height) {
            try {
                bitmap = Bitmap.createBitmap(bitmap, left, top, width, height);
            } catch (Exception e) {
                return null;
            }
        }
        return bitmap;
    }

    /**
     * 图片旋转
     *
     * @param tmpBitmap
     * @param degrees
     * @return
     */
    public static Bitmap rotateToDegrees(Bitmap tmpBitmap, float degrees) {
        Matrix matrix = new Matrix();
        matrix.reset();
        matrix.setRotate(degrees);
        return Bitmap.createBitmap(tmpBitmap, 0, 0,
                tmpBitmap.getWidth(), tmpBitmap.getHeight(), matrix, true);
    }


    /**
     * 摄像头配置
     */
    public void initCameraParams() {

        Log.e("---","initCameraParams");

        stopPreview();
        //获取camera参数
        Camera.Parameters camParams = mCamera.getParameters();
        List<Camera.Size> sizes = camParams.getSupportedPreviewSizes();
        //确定前面定义的预览宽高是camera支持的，不支持取就更大的
        int temp_width = 0,ii=0;
        for (int i = 0; i < sizes.size(); i++) {

            //Log.d("width-height :"+i+"-",sizes.get(i).width+"----"+sizes.get(i).height+"---");

//            if ((sizes.get(i).width >= imageWidth && sizes.get(i).height >= imageHeight) || i == sizes.size() - 1) {
//                imageWidth = sizes.get(i).width;
//                imageHeight = sizes.get(i).height;
//                break;
//            }

            if(sizes.get(i).width>temp_width){
                temp_width = sizes.get(i).width;
                ii=i;
            }

        }

        imageWidth = sizes.get(ii).width;
        imageHeight = sizes.get(ii).height;



        Log.d("width-height:",imageWidth+"----"+imageHeight+"---");

        //设置最终确定的预览大小
        camParams.setPreviewSize(imageWidth, imageHeight);
        //camParams.setPreviewSize(600, 500);
        //设置帧率
        camParams.setPreviewFrameRate(frameRate);
        //启用参数
        mCamera.setParameters(camParams);





        //mCamera.setDisplayOrientation(90);


        Camera.CameraInfo info = new Camera.CameraInfo();
        Camera.getCameraInfo(0, info);
        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + dg) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - dg + 360) % 360;
        }
        mCamera.setDisplayOrientation(result);










        //开始预览
        startPreview();
    }




    /**
     * 开始预览
     */
    public void startPreview() {

        Log.e("---","startPreView");

        try {
            mCamera.setPreviewCallback(this);
            mCamera.setPreviewDisplay(mHolder);//set the surface to be used for live preview
            mCamera.startPreview();
            mCamera.autoFocus(autoFocusCB);
        } catch (IOException e) {
            mCamera.release();
            mCamera = null;
        }
    }

    /**
     * 停止预览
     */
    public void stopPreview() {

        Log.e("---","stopPreview");

        if (mCamera != null) {
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
        }
    }

    /**
     * 打开指定摄像头
     */
    public void openCamera() {

        Log.e("---","openCamera");

        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        Log.e("---","count --"+Camera.getNumberOfCameras());
        for (int cameraId = 0; cameraId < Camera.getNumberOfCameras(); cameraId++) {
            Log.e("---","for --"+cameraId);
            Camera.getCameraInfo(cameraId, cameraInfo);
            Log.e("---","for --"+cameraInfo.facing);
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {

                try {
                    Log.e("---","mCamera Created pre");
                    //mCamera = Camera.open(cameraId);
                    mCamera = Camera.open();
                    Log.e("---","mCamera Created");
                } catch (Exception e) {
                    if (mCamera != null) {
                        mCamera.release();
                        mCamera = null;
                    }
                }

                break;
            }
        }
    }


    /**
     * 摄像头自动聚焦
     */
    Camera.AutoFocusCallback autoFocusCB = new Camera.AutoFocusCallback() {
        public void onAutoFocus(boolean success, Camera camera) {

            //postDelayed(doAutoFocus, 100);

            if(success){
                isDuijiao = true;
            }else{
                postDelayed(doAutoFocus, 100);
            }

        }
    };

    private Runnable doAutoFocus = new Runnable() {
        public void run() {
            if (mCamera != null) {
                try {
                    mCamera.autoFocus(autoFocusCB);
                } catch (Exception e) {
                }
            }
        }
    };

    /**
     * 释放
     */
    public void release() {

        Log.e("---","release");

        if (isPreviewOn && mCamera != null) {
            isPreviewOn = false;
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
        closeCamera();
    }

    public void closeCamera() {

        synchronized (this) {
            try {
                if (mCamera != null) {
                    mCamera.setPreviewCallback(null);
                    mCamera.stopPreview();
                    mCamera.release();
                    mCamera = null;
                }
            } catch (Exception e) {
                Log.i("TAG", e.getMessage());
            }
        }
    }


    private void saveImage(Bitmap bitmap, String saveFile) {
        try {
            FileOutputStream fos = new FileOutputStream(saveFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * 读取图片属性：旋转的角度
     * @param path 图片绝对路径
     * @return degree旋转的角度
     */
    public static int readPictureDegree(String path) {
        int degree  = 0;
        try {

            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }
    /*
     * 旋转图片
     * @param angle
     * @param bitmap
     * @return Bitmap
     */
    public static Bitmap rotaingImageView(int angle , Bitmap bitmap) {
        //旋转图片 动作
        Matrix matrix = new Matrix();;
        matrix.postRotate(angle);
        System.out.println("angle2=" + angle);
        // 创建新的图片
        Bitmap resizedBitmap = Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        return resizedBitmap;
    }


}
