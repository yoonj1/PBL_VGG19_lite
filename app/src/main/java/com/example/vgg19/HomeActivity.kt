package com.example.vgg19

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.widget.Button

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        Log.i("test", "test 0")

        // 버튼에 클릭 리스너 설정
        val button: Button = findViewById(R.id.button)
        button.setOnClickListener {
            // MainActivity로 이동
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
    }
}
