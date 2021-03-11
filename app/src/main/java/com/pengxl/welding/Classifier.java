package com.pengxl.welding;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import com.pengxl.welding.ml.Inference;
import com.pengxl.welding.ml.Model;
import com.pengxl.welding.ml.Transform;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class Classifier {

    private Context context;
    private List<YoloOutput> yoloOutputs;
    private final float[][] anchor_box1 = new float[][] {new float[]{30f / 416, 61f / 416}, new float[]{62f / 416, 45f / 416}, new float[]{59f / 416, 119f / 416}};
    private final float[][] anchor_box2 = new float[][] {new float[]{10f / 416, 13f / 416}, new float[]{16f / 416, 30f / 416}, new float[]{33f / 416, 23f / 416}};

    public static Bitmap displayImage;

    Classifier(Context context) {
        this.context = context;
    }

    public boolean detect(Bitmap inputImage) {
        TensorBuffer input = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(inputImage);
        int crop = Math.min(inputImage.getWidth(), inputImage.getHeight());
        ImageProcessor imageProcessor = new ImageProcessor
                .Builder()
                .add(new ResizeWithCropOrPadOp(crop, crop))
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .build();
        tensorImage = imageProcessor.process(tensorImage);
        ByteBuffer byteBuffer = tensorImage.getBuffer();
        input.loadBuffer(byteBuffer);
        try {
            Transform transform = Transform.newInstance(context);
            Model model = Model.newInstance(context);
            Inference inf = Inference.newInstance(context);

            TensorBuffer transformed_input = transform.process(input).getOutputFeature0AsTensorBuffer();

            Model.Outputs outputs = model.process(transformed_input);
            TensorBuffer grid_32 = outputs.getOutputFeature0AsTensorBuffer();
            TensorBuffer output1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer output2 = outputs.getOutputFeature2AsTensorBuffer();

//            Log.i("pxl", Arrays.toString(output1.getFloatArray()));
            Inference.Outputs infOutput = inf.process(output1, output2);
            extractBoxes(infOutput, 0.5f);
            nonMaxSuppresion(0.5f);
            model.close();
            transform.close();
            inf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (yoloOutputs.size() > 0) {
            Bitmap b = tensorImage.getBitmap();
            Size displaySize = new Size(448, 448);
            Mat m = new Mat();
            Utils.bitmapToMat(b, m);
            Core.rotate(m, m, Core.ROTATE_90_CLOCKWISE);
            for (YoloOutput element: yoloOutputs) {
                Imgproc.rectangle(
                        m,
                        new Point((1 - element.bbox[1]) * 224, element.bbox[2] * 224),
                        new Point((1 - element.bbox[3]) * 224, element.bbox[0] * 224),
                        new Scalar(0, 0, 255),
                        2
                );
            }
            Utils.matToBitmap(m, b);
            int width = b.getWidth();
            int height = b.getHeight();
            float scaleWidth = 2f;
            float scaleHeight = 2f;
            Matrix matrix = new Matrix();
            matrix.postScale(scaleWidth, scaleHeight);
            displayImage = Bitmap.createBitmap(b, 0, 0, width, height, matrix, true);
            return true;
        }
        return false;
    }

    private void extractBoxes(Inference.Outputs infOutput, float scoreThreshold) {
        float[] bbox1 = infOutput.getOutputFeature0AsTensorBuffer().getFloatArray();
        float[] objectScore1 = infOutput.getOutputFeature1AsTensorBuffer().getFloatArray();
        float[] classProb1 = infOutput.getOutputFeature2AsTensorBuffer().getFloatArray();
        float[] predBox1 = infOutput.getOutputFeature3AsTensorBuffer().getFloatArray();
        float[] bbox2 = infOutput.getOutputFeature4AsTensorBuffer().getFloatArray();
        float[] objectScore2 = infOutput.getOutputFeature5AsTensorBuffer().getFloatArray();
        float[] classProb2 = infOutput.getOutputFeature6AsTensorBuffer().getFloatArray();
        float[] predBox2 = infOutput.getOutputFeature7AsTensorBuffer().getFloatArray();
        yoloOutputs = new ArrayList<>();

//        for (int i = 0; i < output1.length; i += 6) {
//            if (MathFunction.sigmoid(output1[i + 4]) > 0.5) {
//                Log.i("pxl", "center: " + MathFunction.sigmoid(output1[i]) + ", " + MathFunction.sigmoid(output1[i + 1]));
//                Log.i("pxl", "grid1: " + (i / 18) % 13 + ", " + (i / 234) % 13);
//                float[] bbox = getBbox(output1[i], output1[i + 1], output1[i + 2], output1[i + 3],
//                        (i / 18) % 13, (i / 234) % 13, 14, this.anchor_box1[(i / 6) % 3]);
//                yoloOutputs.add(new YoloOutput(
//                        bbox,
//                        output1[i + 4],
//                        output1[i + 5],
//                        this.anchor_box1[(i / 6) % 3]));
//            }
//        }
//        for (int i = 0; i < output2.length; i += 6) {
//            if (MathFunction.sigmoid(output2[i + 4]) > 0.5) {
//                Log.i("pxl", "center: " + MathFunction.sigmoid(output2[i]) + ", " + MathFunction.sigmoid(output2[i + 1]));
//                Log.i("pxl", "grid2: " + (i / 18) % 13 + ", " + (i / 234) % 13);
//                float[] bbox = getBbox(output2[i], output2[i + 1], output2[i + 2], output2[i + 3],
//                        (i / 18) % 13, (i / 234) % 13, 28, this.anchor_box2[(i / 6) % 3]);
//                yoloOutputs.add(new YoloOutput(
//                        bbox,
//                        output2[i + 4],
//                        output2[i + 5],
//                        this.anchor_box2[(i / 6) % 3]));
//            }
//        }

        for (int i = 0; i < objectScore1.length; i++) {
            if (objectScore1[i] > 0.5) {
                yoloOutputs.add(new YoloOutput(
                        new float[]{bbox1[4 * i], bbox1[4 * i + 1], bbox1[4 * i + 2], bbox1[4 * i + 3]},
                        objectScore1[i],
                        classProb1[i],
                        anchor_box1[i % 3]));
            }
        }
        for (int i = 0; i < objectScore2.length; i++) {
            if (objectScore2[i] > 0.5) {
                yoloOutputs.add(new YoloOutput(
                        new float[]{bbox2[4 * i], bbox2[4 * i + 1], bbox2[4 * i + 2], bbox2[4 * i + 3]},
                        objectScore2[i],
                        classProb2[i],
                        anchor_box2[i % 3]));
            }
        }
    }

    private float[] getBbox(float x, float y, float w, float h, int gridX, int gridY, int gridSize, float[] anchor) {
        x = MathFunction.sigmoid(x);
        y = MathFunction.sigmoid(y);
        w = MathFunction.sigmoid(w);
        h = MathFunction.sigmoid(h);
        x = (x + (float) gridX) / gridSize;
        y = (y + (float) gridY) / gridSize;
        w = (float) Math.exp(Double.parseDouble(String.valueOf(w))) * anchor[0];
        h = (float) Math.exp(Double.parseDouble(String.valueOf(h))) * anchor[1];
        float x1 = x - w / 2;
        float y1 = y - h / 2;
        float x2 = x + w / 2;
        float y2 = y + h / 2;
        return new float[]{x1, y1, x2, y2};
    }

    private void nonMaxSuppresion(float IoUThreshold) {
        yoloOutputs.sort(new Comparator<YoloOutput>() {
            @Override
            public int compare(YoloOutput o1, YoloOutput o2) {
                return Float.compare(o2.objScore, o1.objScore);
            }
        });
        ArrayList<YoloOutput> result = new ArrayList<>();
        for (YoloOutput element: yoloOutputs) {
            if (result.size() == 0) {
                result.add(element);
                continue;
            }
            int flag = 0;
            for (YoloOutput x: result) {
                if (Float.compare(calcIoU(x, element), IoUThreshold) == 1) {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0) {
                result.add(element);
            }
        }
        yoloOutputs = result;
    }

    private float calcIoU(YoloOutput x, YoloOutput y) {
        float intW = Math.max(Math.min(x.bbox[2], y.bbox[2]) - Math.max(x.bbox[0], y.bbox[0]), 0f);
        float intH = Math.max(Math.min(x.bbox[3], y.bbox[3]) - Math.max(x.bbox[1], y.bbox[1]), 0f);
        float intArea = intW * intH;
        float xArea = (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]);
        float yArea = (y.bbox[2] - y.bbox[0]) * (y.bbox[3] - y.bbox[1]);
        return intArea / (xArea + yArea - intArea);
    }

    private void saveImage(TensorImage tensorImage) {
        Bitmap b = tensorImage.getBitmap();
        Mat m = new Mat();
        Utils.bitmapToMat(b, m);
        for (YoloOutput element: yoloOutputs) {
            Imgproc.rectangle(
                    m,
                    new Point(element.bbox[1] * 224, (1 - element.bbox[2]) * 224),
                    new Point(element.bbox[3] * 224, (1 - element.bbox[0]) * 224),
                    new Scalar(0, 0, 255),
                    2
            );
        }
        Utils.matToBitmap(m, b);
        try {
            ContentValues contentValues = new ContentValues();
            contentValues.put(MediaStore.Images.Media.DISPLAY_NAME, "welding.jpg");
            contentValues.put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/signImage");
            contentValues.put(MediaStore.Images.Media.MIME_TYPE, "image/JPEG");
            Uri uri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
            if (uri != null) {
                OutputStream outputStream = context.getContentResolver().openOutputStream(uri);
                if (outputStream != null) {
                    b.compress(Bitmap.CompressFormat.JPEG, 90, outputStream);
                    outputStream.flush();
                    outputStream.close();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static class YoloOutput {
        float[] bbox;
        float[] anchor;
        float objScore;
        float classProb;

        YoloOutput(float[] bbox, float objScore, float classProb, float[] anchor) {
            this.bbox = bbox;
            this.objScore = objScore;
            this.classProb = classProb;
            this.anchor = anchor;
        }
    }
}
