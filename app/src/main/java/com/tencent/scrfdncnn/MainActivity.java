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

package com.tencent.scrfdncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat;
import android.widget.Toast;
import android.media.RingtoneManager;
import android.net.Uri;
import android.media.Ringtone;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;


public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private SCRFDNcnn scrfdncnn = new SCRFDNcnn();
    private int facing = 0;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;


    ArrayList<String> paths = null;
    ArrayList<String> names= null;
    List<Map<String, Object>> listItems;

    public int PlayNotice(){
        Uri uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
        Ringtone rt = RingtoneManager.getRingtone(getApplicationContext(), uri);
        rt.play();
        return 0;
    }

    void GetImagesPath(){

        paths = new ArrayList();
        names = new ArrayList();

        Cursor cursor = getContentResolver().query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, null, null, null, null);
        int index = 0;
        while (cursor.moveToNext() && index++ < 10) {
            //获取图片的名称
            String name = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME));
            // 获取图片的绝对路径
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            String path = cursor.getString(column_index);


            paths.add(path);
            names.add(name);

            Log.i("GetImagesPath", "GetImagesPath: name = "+name+"  path = "+ path);


        }
        listItems = new ArrayList<>();
        for (int i = 0; i < paths.size(); i++) {
            Map<String, Object> map = new HashMap<>();
            map.put("name", paths.get(i));
            map.put("desc", names.get(i));
            listItems.add(map);
        }
    }

    public static byte[] fileToByteArray(String filePath) {
        //1.创建源yu目的地
        File file = new File(filePath);
        byte[] ds = null;
        //选择流
        InputStream zp = null;
        ByteArrayOutputStream boos = null;
        boos = new ByteArrayOutputStream();
        try {
            zp = new FileInputStream(file);
            byte[] frush = new byte[1024];//1024表示1k为一段
            int len = -1;
            while((len=zp.read(frush))!=-1) {
                boos.write(frush,0,len);//写出到字节数组中

            }
            boos.flush();
            return boos.toByteArray();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }finally {
            if(zp!=null) {
                try {
                    zp.close();
                } catch (IOException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

        }
        return null;
    }


    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();

        } catch (IOException ex) {
            Log.i("Info", "Failed to upload a file");
        }
        return "";
    }

    private static boolean hasText(String str) {
        return str != null && !str.isEmpty() && containsText(str);
    }

    private static boolean containsText(String str) {
        int strLen = str.length();
        for (int i = 0; i < strLen; i++) {
            if (!Character.isWhitespace(str.charAt(i))) {
                return true;
            }
        }
        return false;
    }

    private void sort(int[][] ret) {
        Arrays.sort(ret, (o1, o2) -> {
            if (Math.abs(o1[1] - o2[1]) < o1[3]) {
                return o1[0] - o2[0];
            } else {
                return o1[1] - o2[1];
            }
        });
        int index = 1;
//        for (int[] r : ret) {
//            if (r[4] == 0) {
//                continue;
//            }
//            r[4] = i++;
//        }
        for (int i = 0; i < ret.length; i++) {
            if (ret[i][4] == 0) {
                continue;
            }
            //前后两个框重叠则滤除一个
            if (i >= 1 && ret[i - 1][4] != 0 && Math.abs(ret[i][0] - ret[i - 1][0]) < ret[i][2] && Math.abs(ret[i][1] - ret[i - 1][1]) < ret[i][3]) {
                continue;
            }
            ret[i][4] = index++;
        }

    }

    private int[][] convertStringToArray(String str) {
        if (!hasText(str)) {
            //throw new RuntimeException();
            return new int[0][0];
        }
        String[] strs = str.split(" ");
        int n = strs.length;
        int[][] ret = new int[n][5];
        for (int i = 0; i < n; i++) {
            String[] tempStr = strs[i].split("_");
            if (tempStr.length < 5) {
                break;
            }
            for (int j = 0; j < 5; j++) {
                ret[i][j] = Integer.valueOf(tempStr[j]);
            }
        }
        //先按y排，再按x排
        sort(ret);

        return ret;
    }

    private String ArrayToString(int[][] ret) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < ret.length; i++) {
            for (int j = 0; j < ret[0].length; j++) {
                sb.append(ret[i][j] + " ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.main);

        GetImagesPath();

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);
        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        Button  buttonSnap =(Button)findViewById(R.id.buttonSnap);

        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                scrfdncnn.closeCamera();

                scrfdncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        buttonSnap.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                scrfdncnn.HandSnap();
                //byte[] data = null;
                //data = fileToByteArray(paths.get(0));

                //scrfdncnn.RecImg(data, 1920, 1080);
                Toast.makeText(getApplicationContext(), "snap suc" , Toast.LENGTH_SHORT).show();
//                System.out.println("Path:" + scrfdncnn.getSnapRoiImg());
//                Toast.makeText(getApplicationContext(), "Path:" + scrfdncnn.getSnapRoiImg()
//                        , Toast.LENGTH_SHORT).show();
                int[][] ret = convertStringToArray(scrfdncnn.getSnapRoiImg());
                Toast.makeText(getApplicationContext(), "Path:" + ArrayToString(ret)
                        , Toast.LENGTH_SHORT).show();
                PlayNotice();
            }
        });




        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();
    }

    private void reload()
    {

        boolean ret_init = scrfdncnn.loadModel(getAssets(),  current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "scrfdncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        scrfdncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        scrfdncnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        scrfdncnn.closeCamera();
    }
}
