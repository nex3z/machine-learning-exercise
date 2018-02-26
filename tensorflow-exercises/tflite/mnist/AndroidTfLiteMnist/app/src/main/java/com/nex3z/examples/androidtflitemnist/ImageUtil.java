package com.nex3z.examples.androidtflitemnist;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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


}
