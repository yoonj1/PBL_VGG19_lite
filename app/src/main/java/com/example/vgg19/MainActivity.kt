package com.example.vgg19

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.lang.ref.WeakReference
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private var audioUri: Uri? = null
    private var mfccImageUri: Uri? = null // 갤러리에 저장된 이미지 URI

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val selectAudioButton = findViewById<Button>(R.id.select_audio_button)
        selectAudioButton.setOnClickListener { openAudioFilePicker() }

        val extractMfccButton = findViewById<Button>(R.id.extract_mfcc_button)
        extractMfccButton.setOnClickListener { extractMfcc() }

        val viewMfccImageButton = findViewById<Button>(R.id.view_mfcc_image_button)
        viewMfccImageButton.setOnClickListener { displayMfccImageInGallery() }

        val classifyButton = findViewById<Button>(R.id.classify_button)
        classifyButton.setOnClickListener { classifyImage() }
    }

    private fun openAudioFilePicker() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "audio/*"
        }
        startActivityForResult(intent, PICK_AUDIO_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_AUDIO_REQUEST && resultCode == RESULT_OK && data != null) {
            audioUri = data.data
            Toast.makeText(this, "음성 파일 선택 완료", Toast.LENGTH_SHORT).show()
        }
    }

    private fun extractMfcc() {
        if (audioUri != null) {
            MFCCExtractionTask(this).execute(audioUri)
        } else {
            Toast.makeText(this, "먼저 음성 파일을 선택하세요.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun classifyImage() {
        val resultTextView = findViewById<TextView>(R.id.resultTextView)
        // 더미 데이터를 이용해 "진짜입니다" 또는 "가짜입니다"로 결과 표시
        val dummyPrediction = (0..1).random() // 0: 진짜, 1: 가짜
        resultTextView.text = if (dummyPrediction == 0) "진짜입니다" else "가짜입니다"
    }

    // 갤러리에 MFCC 이미지를 저장하는 함수
    private fun saveMfccImageToGallery(bitmap: Bitmap, context: Context): Uri? {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "mfcc_image_${System.currentTimeMillis()}.png")
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MFCCImages")
        }

        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        uri?.let {
            context.contentResolver.openOutputStream(it)?.use { outStream ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream)
                outStream.flush()
            }
        }
        return uri
    }

    private inner class MFCCExtractionTask(context: Context) : AsyncTask<Uri, Void, Array<FloatArray>?>() {
        private val contextRef = WeakReference(context)

        override fun doInBackground(vararg uris: Uri?): Array<FloatArray>? {
            val uri = uris[0] ?: return null
            var mfccResult: Array<FloatArray>? = null
            try {
                val inputStream = contentResolver.openInputStream(uri) ?: return null
                val audioData = readAudioData(inputStream)
                inputStream.close()

                if (audioData != null) {
                    val mfcc = MFCC()
                    val mfccFlatArray = mfcc.computeMFCC(audioData)
                    val numFrames = mfccFlatArray.size / MFCC.MEL_BANDS
                    mfccResult = Array(numFrames) { FloatArray(MFCC.MEL_BANDS) }
                    for (i in 0 until numFrames) {
                        System.arraycopy(mfccFlatArray, i * MFCC.MEL_BANDS, mfccResult[i], 0, MFCC.MEL_BANDS)
                    }
                }
            } catch (e: Exception) {
                Log.e("MFCC Extraction Error", "Error extracting MFCC: ", e)
            }
            return mfccResult
        }

        override fun onPostExecute(mfccResult: Array<FloatArray>?) {
            val context = contextRef.get()
            if (mfccResult != null && context != null) {
                val bitmap = createMfccImage(mfccResult, context)
                if (bitmap != null) {
                    mfccImageUri = saveMfccImageToGallery(bitmap, context)
                    if (mfccImageUri != null) {
                        Toast.makeText(context, "MFCC 이미지가 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(context, "MFCC 이미지 저장 실패", Toast.LENGTH_SHORT).show()
                    }
                }
            } else {
                Toast.makeText(context, "MFCC 추출 실패", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun displayMfccImageInGallery() {
        mfccImageUri?.let {
            val intent = Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(it, "image/*")
                flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
            }
            startActivity(intent)
        } ?: Toast.makeText(this, "갤러리에 저장된 이미지가 없습니다.", Toast.LENGTH_SHORT).show()
    }

    private fun createMfccImage(mfccArray: Array<FloatArray>, context: Context): Bitmap? {
        val width = mfccArray.size
        val height = mfccArray[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // 데이터 범위를 로그 변환 및 정규화하여 시각적 표현 향상
        val logMfccArray = mfccArray.map { row ->
            row.map { value -> log10(1 + max(1e-10f, value)) }.toFloatArray()
        }.toTypedArray()

        // Array<Array<Float>>를 List로 변환한 후 flatten()을 호출
        val minVal = logMfccArray.flatMap { it.asList() }.minOrNull() ?: 0f
        val maxVal = logMfccArray.flatMap { it.asList() }.maxOrNull() ?: 1f

        for (x in logMfccArray.indices) {
            for (y in logMfccArray[x].indices) {
                val normalizedValue = (logMfccArray[x][y] - minVal) / (maxVal - minVal)
                val color = Color.rgb(
                    (normalizedValue * 255).toInt(),
                    (normalizedValue * 128).toInt(),
                    (255 - normalizedValue * 255).toInt()
                )
                bitmap.setPixel(x, y, color)
            }
        }
        return bitmap
    }

    private fun readAudioData(inputStream: InputStream): FloatArray? {
        return try {
            val byteBuffer = ByteArrayOutputStream()
            val bufferSize = 1024
            val buffer = ByteArray(bufferSize)

            var len: Int
            while (inputStream.read(buffer).also { len = it } != -1) {
                byteBuffer.write(buffer, 0, len)
            }
            val audioBytes = byteBuffer.toByteArray()

            val audioData = FloatArray(audioBytes.size / 2)
            for (i in audioData.indices) {
                audioData[i] =
                    ((audioBytes[i * 2].toInt() and 0xFF) or (audioBytes[i * 2 + 1].toInt() shl 8)) / 32768.0f
            }
            audioData
        } catch (e: Exception) {
            Log.e("Audio Data Read Error", "Error reading audio data", e)
            null
        }
    }

    companion object {
        private const val PICK_AUDIO_REQUEST = 1
    }
}
