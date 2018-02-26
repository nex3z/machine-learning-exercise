package com.nex3z.examples.androidtflitemnist;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();

    private TextView mTvOutput;
    private TextView mTvPredict;
    private TfLiteModel mTfLiteModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init();
    }

    private void init() {
        initView();
        initModel();
    }

    private void initView() {
        mTvOutput = findViewById(R.id.tv_output);
        mTvPredict = findViewById(R.id.tv_predict);
        ImageView ivInput = findViewById(R.id.iv_input);
        Button btnRun = findViewById(R.id.btn_run);

        final Bitmap input = ImageUtil.getImageAsset(this, "five.png");
        ivInput.setImageBitmap(input);

        btnRun.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                run(input);
            }
        });
    }

    private void initModel() {
        try {
            mTfLiteModel = new TfLiteModel(this);
        } catch (IOException e) {
            Log.e(LOG_TAG, "init(): Failed to create tflite model", e);
        }
    }

    private void run(Bitmap input) {
        if (mTfLiteModel == null) {
            Log.e(LOG_TAG, "run(): Model is not initialized, abort");
            return;
        }
        float[] output = mTfLiteModel.apply(input);
        mTvOutput.setText(Arrays.toString(output));
        mTvPredict.setText(String.valueOf(getMaxIdx(output)));
    }

    private int getMaxIdx(float[] values) {
        float max = -1;
        int maxIdx = -1;;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > max) {
                max = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
