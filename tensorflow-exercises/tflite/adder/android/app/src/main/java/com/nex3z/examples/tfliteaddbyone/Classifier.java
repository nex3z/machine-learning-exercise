package com.nex3z.examples.tfliteaddbyone;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class Classifier {
    private static final String LOG_TAG = Classifier.class.getSimpleName();
    private static final String MODEL_PATH = "add_by_one.tflite";

    private Interpreter tflite;

    Classifier(Activity activity) throws IOException {
        tflite = new Interpreter(loadModelFile(activity));
    }

    public void close() {
        tflite.close();
        tflite = null;
    }

    public float compute(float input) {
        float[] result = new float[1];
        float[] x = { input };
        tflite.run(x, result);
        Log.i(LOG_TAG, "x = " + Arrays.toString(x) + ", result = " + Arrays.toString(result));
        return result[0];
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

}
