package com.example.biopredict;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    // EditText fields for inputs
    EditText inputFieldisMale;
    EditText inputFieldisBlack;
    EditText inputFieldisSmoker;
    EditText inputFieldisDiabetic;
    EditText inputFieldisHypertensive;
    EditText inputFieldAge;
    EditText inputFieldSystolic;
    EditText inputFieldCholesterol;
    EditText inputFieldHDL;

    Button predictBtn;
    TextView resultTV;

    Interpreter interpreter;

    // Min and max values for normalization
    float[] minValues = {0, 0, 0, 0, 0, 20, 90, 130, 20};  //
    float[] maxValues = {1, 1, 1, 1, 1, 79, 200, 320, 100};  //

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Load the TensorFlow Lite model
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Initialize EditText fields
        inputFieldisMale = findViewById(R.id.editTextSex);
        inputFieldisBlack = findViewById(R.id.editTextRace);
        inputFieldisSmoker = findViewById(R.id.editTextSmoker);
        inputFieldisDiabetic = findViewById(R.id.editTextDiabetic);
        inputFieldisHypertensive = findViewById(R.id.editTextHypertensive);
        inputFieldAge = findViewById(R.id.editTextAge);
        inputFieldSystolic = findViewById(R.id.editTextSystolicPressure);
        inputFieldCholesterol = findViewById(R.id.editTextCholesterol);
        inputFieldHDL = findViewById(R.id.editTextHDLCholesterol);

        // Initialize Button and TextView for results
        predictBtn = findViewById(R.id.buttonPredict);
        resultTV = findViewById(R.id.textViewRisk);

        // Set onClickListener for the predict button
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {
                // Collect input values
                float[][] inputs = new float[1][9]; // 1 sample, 9 features
                inputs[0][0] = Float.parseFloat(inputFieldisMale.getText().toString());
                inputs[0][1] = Float.parseFloat(inputFieldisBlack.getText().toString());
                inputs[0][2] = Float.parseFloat(inputFieldisSmoker.getText().toString());
                inputs[0][3] = Float.parseFloat(inputFieldisDiabetic.getText().toString());
                inputs[0][4] = Float.parseFloat(inputFieldisHypertensive.getText().toString());
                inputs[0][5] = Float.parseFloat(inputFieldAge.getText().toString());
                inputs[0][6] = Float.parseFloat(inputFieldSystolic.getText().toString());
                inputs[0][7] = Float.parseFloat(inputFieldCholesterol.getText().toString());
                inputs[0][8] = Float.parseFloat(inputFieldHDL.getText().toString());

                // Normalize inputs
                float[][] normalizedInputs = new float[1][9];
                for (int i = 0; i < 9; i++) {
                    normalizedInputs[0][i] = normalize(inputs[0][i], minValues[i], maxValues[i]);
                }

                // Run inference
                float[][] output = new float[1][1]; // Single output for the prediction
                interpreter.run(normalizedInputs, output);

                // Display the result (you can scale back the prediction if needed)
                resultTV.setText("Risk: " + (output[0][0] * 100)); // Scale prediction back to percentage
            }
        });
    }

    // Normalization function
    private float normalize(float value, float minVal, float maxVal) {
        return (value - minVal) / (maxVal - minVal);
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try (AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("regression_AQ.tflite");
             FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = fileInputStream.getChannel();
            long startOffset = assetFileDescriptor.getStartOffset();
            long length = assetFileDescriptor.getLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
        }
    }
}

