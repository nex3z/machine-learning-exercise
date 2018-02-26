package com.nex3z.examples.tfliteimage;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TfLiteModel {
    private static final String LOG_TAG = TfLiteModel.class.getSimpleName();

    // private static final String MODEL_PATH = "conv.tflite";
    private static final String MODEL_PATH = "conv_var_input.tflite";

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_HEIGHT = 150;
    private static final int DIM_IMG_SIZE_WIDTH = 200;
    private static final int DIM_PIXEL_SIZE = 3;

    private Interpreter mTfLite;
    private ByteBuffer mImgData = null;
    private int[] mImagePixels = new int[DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH];
    private float[][][][] mProcessed = new float[1][DIM_IMG_SIZE_HEIGHT][DIM_IMG_SIZE_WIDTH][DIM_PIXEL_SIZE];


    TfLiteModel(Activity activity) throws IOException {
        mTfLite = new Interpreter(loadModelFile(activity));

        mImgData = ByteBuffer.allocateDirect(
                        4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_HEIGHT * DIM_IMG_SIZE_WIDTH * DIM_PIXEL_SIZE);
        mImgData.order(ByteOrder.nativeOrder());
    }

    public Bitmap apply(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);
        mTfLite.run(mImgData, mProcessed);
        // ImageUtil.printImageArray(mProcessed[0], DIM_IMG_SIZE_HEIGHT, DIM_IMG_SIZE_WIDTH);
        return ImageUtil.createBitmap(mProcessed[0], DIM_IMG_SIZE_HEIGHT, DIM_IMG_SIZE_WIDTH);
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImgData == null) {
            return;
        }
        mImgData.rewind();

        bitmap.getPixels(mImagePixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_WIDTH; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_HEIGHT; ++j) {
                final int val = mImagePixels[pixel++];
                mImgData.putFloat((val >> 16) & 0xFF);
                mImgData.putFloat((val >> 8) & 0xFF);
                mImgData.putFloat((val) & 0xFF);
            }
        }
    }
}
