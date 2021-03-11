package com.pengxl.welding;

import androidx.appcompat.app.AppCompatActivity;

import android.media.Image;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

public class DisplayActivity extends AppCompatActivity {

    private Button back;
    private ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_display);

        back = (Button) findViewById(R.id.display_back);
        imageView = (ImageView) findViewById(R.id.display_image);

        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish();
            }
        });
        imageView.setImageBitmap(Classifier.displayImage);
    }
}