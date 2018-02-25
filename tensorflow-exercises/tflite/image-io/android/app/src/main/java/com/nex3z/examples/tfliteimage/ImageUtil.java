package com.nex3z.examples.tfliteimage;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;

public class ImageUtil {
    private static final String LOG_TAG = ImageUtil.class.getSimpleName();

    private ImageUtil() {}

    public static Bitmap getImageAsset(Context context, String fileName) {
        InputStream is = null;
        Bitmap bitmap = null;
        try {
            is = context.getAssets().open(fileName);
            bitmap = BitmapFactory.decodeStream(is);
        } catch (IOException exception) {
            Log.e(LOG_TAG, "getImageAsset(): ", exception);
        } finally {
            try {
                if (is != null) {
                    is.close();
                }
            } catch (IOException closeException) {
                Log.e(LOG_TAG, "getImageAsset(): ", closeException);
            }

        }
        return bitmap;
    }

    public static Bitmap createBitmap(float[][][] data, int height, int width) {
        int[] image = new int[height * width * 3];
        int pixel = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int color = getColor(data[h][w]);
                image[pixel++] = color;
            }
        }
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.setPixels(image, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private static int getColor(float[] data) {
         return Color.argb(255, (int) data[0], (int) data[1], (int) data[2]);
//        return Color.argb(255, (int) data[2], (int) data[1], (int) data[0]);
    }

    @SuppressLint("DefaultLocale")
    public static void printImageArray(float[][][] data, int height, int width) {
        for (int h = 0; h < height; h++) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int w = 0; w < width; w++) {
                sb.append(String.format("(%3d,%3d,%3d), ",
                        (int) data[h][w][0], (int) data[h][w][1], (int) data[h][w][2]));
            }
            sb.append("]");
            Log.i(LOG_TAG, "printImageArray(): h = " + h + ", " + sb.toString());
        }
    }

}
