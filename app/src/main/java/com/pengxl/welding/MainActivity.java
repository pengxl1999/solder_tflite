package com.pengxl.welding;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

import static org.opencv.core.Core.FILLED;


public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getName();
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final int PERMISSIONS_REQUEST = 1;
    private final Object syncObject = new Object();
    private JavaCamera2View cameraView;
    private Classifier classifier;
    private Mat currentFrame, outputFrame;
    private boolean isProcessing = false;

    static {
        if (!OpenCVLoader.initDebug()) {
            // Error
            Log.e(TAG, "OpenCV library is not found!");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        classifier = new Classifier(MainActivity.this);
        cameraView = (JavaCamera2View) findViewById(R.id.opencv_camera);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV library is not found!");
        }
        if (hasPermission()) {
            cameraView.setVisibility(SurfaceView.VISIBLE);
            cameraView.enableView();
            cameraView.setCvCameraViewListener(this);
        } else {
            requestPermission();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        currentFrame = new Mat();
        outputFrame = new Mat();
        Log.i(TAG, "onCameraViewStarted");
    }

    @Override
    public void onCameraViewStopped() {
        Log.i(TAG, "onCameraViewStopped");
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (inputFrame == null) {
            return null;
        }
        if (!isProcessing) {
            isProcessing = true;
            synchronized (syncObject) {
                outputFrame = inputFrame.rgba();
                Imgproc.cvtColor(inputFrame.gray(), currentFrame, Imgproc.COLOR_GRAY2RGBA, 4);
                processImage(currentFrame);
                isProcessing = false;
            }
            return outputFrame;
        }
        return null;
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST) {
            if (allPermissionsGranted(grantResults)) {
                cameraView.setVisibility(SurfaceView.VISIBLE);
                cameraView.enableView();
                cameraView.setCvCameraViewListener(this);
            } else {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private boolean hasPermission() {
        return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {
        Log.i(TAG, "pxl");
        if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
            Toast.makeText(
                    MainActivity.this,
                    "Camera permission is required for this demo",
                    Toast.LENGTH_LONG)
                    .show();
        }
        requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }

    private void processImage(Mat image) {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, bitmap);
        if (classifier.detect(bitmap)) {
            Intent intent = new Intent(MainActivity.this, DisplayActivity.class);
            startActivity(intent);
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}