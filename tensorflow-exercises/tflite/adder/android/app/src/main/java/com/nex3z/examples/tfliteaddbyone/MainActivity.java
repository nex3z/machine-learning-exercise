package com.nex3z.examples.tfliteaddbyone;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    private EditText mEtInput;
    private Button mBtnCompute;
    private TextView mTvResult;
    private Classifier mClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mEtInput = findViewById(R.id.et_input);
        mBtnCompute = findViewById(R.id.btn_compute);
        mTvResult = findViewById(R.id.tv_result);
        init();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mClassifier.close();
    }

    private void init() {
        try {
            mClassifier = new Classifier(this);
        } catch (IOException e) {
            Log.e(LOG_TAG, "init(): ", e);
        }
        mBtnCompute.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mClassifier != null) {
                    String input = mEtInput.getText().toString();
                    float result = mClassifier.compute(Float.parseFloat(input));
                    mTvResult.setText(String.valueOf(result));
                }
            }
        });
    }
}
