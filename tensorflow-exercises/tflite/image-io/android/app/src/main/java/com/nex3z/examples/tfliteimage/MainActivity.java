package com.nex3z.examples.tfliteimage;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();

    private ImageView mIvOutput;
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
        mIvOutput = findViewById(R.id.iv_output);
        ImageView ivInput = findViewById(R.id.iv_input);
        Button btnRun = findViewById(R.id.btn_run);

        final Bitmap input = ImageUtil.getImageAsset(this, "iris.jpg");
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
        Bitmap result = mTfLiteModel.apply(input);
        mIvOutput.setImageBitmap(result);
    }
}
