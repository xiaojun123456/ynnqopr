package com.ynnqo.pr;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.widget.Button;

public class StartActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);

        Button start = (Button)findViewById(R.id.btn_start);
        start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent it = new Intent(StartActivity.this,VideoActivity.class);
                startActivity(it);
            }
        });


        //因为谷歌在安卓6.0以后增加了运行时权限功能，即在程序运行时需要用户手动授权，才可以正常运行。
//        if (ContextCompat.checkSelfPermission(StartActivity.this,
//                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
//            ActivityCompat.requestPermissions(StartActivity.this,
//                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
//        }


    }
}
