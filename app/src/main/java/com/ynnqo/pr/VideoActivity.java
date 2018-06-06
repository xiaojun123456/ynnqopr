package com.ynnqo.pr;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.Surface;
import android.view.Window;
import android.view.WindowManager;


public class VideoActivity extends Activity {

    CameraView cv;

    //ImageView iv;

    CropImageView2 ivCrop;

    int dg;


    class bdr extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if(action.equals("dd")){
                String path = intent.getStringExtra("path");
                Bitmap bitmap = BitmapFactory.decodeFile(path);
                //iv.setImageBitmap(bitmap);

                //ivCrop.setImageToCrop(bitmap);
                ivCrop.setImageToCrop2(bitmap);
            }

        }
    }
    private bdr m_bdr = new bdr();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        IntentFilter myIntentFilter = new IntentFilter();
        myIntentFilter.addAction("dd");
        registerReceiver(m_bdr, myIntentFilter);

        Window window = getWindow();

        //隐藏标题栏
        //requestWindowFeature(Window.FEATURE_NO_TITLE);

        //隐藏状态栏
        //定义全屏参数
        int flag= WindowManager.LayoutParams.FLAG_FULLSCREEN;

        //设置当前窗体为全屏显示
        window.setFlags(flag, flag);

        setContentView(R.layout.activity_video);

        dg = getCameraDisplayOrientation();
        cv = (CameraView)findViewById(R.id.main_camera);
        cv.setDG(dg);

        //iv = (ImageView)findViewById(R.id.iv_img);



        ivCrop = (CropImageView2) findViewById(R.id.iv_crop);

        //cv.setTag(iv);

//        new Thread(new Runnable(){
//            @Override
//            public void run() {
//
//
//                while(true) {
//                    try {
//
//
//                        int kk = getCameraDisplayOrientation();
//                        if (kk != dg) {
//
//                            dg = kk;
//                            cv.setDG(dg);
//
//                        }
//
//                        Log.e("dg:", dg + "------------------");
//
//                        Thread.sleep(500);
//
//                    } catch (Exception ex) {
//                    }
//                }
//
//
//            }
//        }).start();

    }

    @Override
    protected void onDestroy(){

        cv.release();
        unregisterReceiver(m_bdr);

        super.onDestroy();
    }

    public int getCameraDisplayOrientation() {
        int rotation =  getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0: degrees = 0; break;
            case Surface.ROTATION_90: degrees = 90; break;
            case Surface.ROTATION_180: degrees = 180; break;
            case Surface.ROTATION_270: degrees = 270; break;
        }
        return  degrees;
    }






}
