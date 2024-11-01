package com.example.vgg19

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.AsyncTask
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.vgg19.R
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    private var audioUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val selectAudioButton = findViewById<Button>(R.id.select_audio_button)
        selectAudioButton.setOnClickListener { openAudioFilePicker() }

        val extractMfccButton = findViewById<Button>(R.id.extract_mfcc_button)
        extractMfccButton.setOnClickListener { extractMfcc() }
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

    private fun createMfccImage(mfccArray: Array<FloatArray>, context: Context): File? {
        val width = mfccArray.size
        val height = mfccArray[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Flatten을 사용하지 않고 min 및 max 값 계산
        var min = Float.MAX_VALUE
        var max = Float.MIN_VALUE
        for (row in mfccArray) {
            for (value in row) {
                if (value < min) min = value
                if (value > max) max = value
            }
        }

        // MFCC 값을 픽셀로 변환
        for (x in mfccArray.indices) {
            for (y in mfccArray[x].indices) {
                val normalizedValue = (mfccArray[x][y] - min) / (max - min)
                val color = Color.rgb((normalizedValue * 255).toInt(), 0, (255 - normalizedValue * 255).toInt())
                bitmap.setPixel(x, y, color)
            }
        }

        return try {
            val file = File(context.cacheDir, "mfcc_image.png")
            FileOutputStream(file).use { outStream ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, outStream)
            }
            file
        } catch (e: Exception) {
            Log.e("MFCC Image Creation", "Failed to save image", e)
            null
        }
    }

    private inner class MFCCExtractionTask(val context: Context) : AsyncTask<Uri, Void, Array<FloatArray>?>() {
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
            if (mfccResult != null) {
                val imageFile = createMfccImage(mfccResult, context)
                if (imageFile != null) {
                    Log.d("MFCC Image", "MFCC image saved at: ${imageFile.absolutePath}")
                    Toast.makeText(this@MainActivity, "MFCC 이미지 생성 완료", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "MFCC 이미지 생성 실패", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this@MainActivity, "MFCC 추출 실패", Toast.LENGTH_SHORT).show()
            }
        }
    }

    companion object {
        private const val PICK_AUDIO_REQUEST = 1
    }
}
