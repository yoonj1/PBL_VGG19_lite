package com.example.vgg19

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private var audioUri: Uri? = null
    private var mfccArray: Array<FloatArray>? = null
    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 백그라운드에서 모델 로드 시도
        CoroutineScope(Dispatchers.IO).launch {
            loadAndInitializeInterpreter()
        }

        findViewById<Button>(R.id.select_audio_button).setOnClickListener { openAudioFilePicker() }
        findViewById<Button>(R.id.extract_mfcc_button).setOnClickListener { extractMfcc() }
        findViewById<Button>(R.id.classify_button).setOnClickListener { classifyAudio() }
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
            CoroutineScope(Dispatchers.IO).launch {
                val mfccResult = extractMfccData(audioUri!!)
                withContext(Dispatchers.Main) {
                    mfccArray = mfccResult
                    Toast.makeText(
                        this@MainActivity,
                        if (mfccResult != null) "MFCC 데이터 추출 완료" else "MFCC 추출 실패",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        } else {
            Toast.makeText(this, "먼저 음성 파일을 선택하세요.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun classifyAudio() {
        val resultTextView = findViewById<TextView>(R.id.resultTextView)

        if (interpreter == null) {
            Toast.makeText(this, "모델을 로드하지 못했습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        mfccArray?.let { mfccData ->
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val isReal = runModelInference(mfccData)
                    withContext(Dispatchers.Main) {
                        resultTextView.text = if (isReal) "진짜입니다" else "가짜입니다"
                    }
                } catch (e: Exception) {
                    Log.e("Inference Error", "Error during model inference", e)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "모델 추론 중 오류 발생", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } ?: Toast.makeText(this, "MFCC 데이터를 먼저 추출하세요.", Toast.LENGTH_SHORT).show()
    }

    private suspend fun runModelInference(mfccArray: Array<FloatArray>): Boolean {
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: intArrayOf(1, 384, 216)
        val inputBuffer = convertMfccToByteBuffer(mfccArray, inputShape)
        val outputArray = Array(1) { FloatArray(1) }
        interpreter?.run(inputBuffer, outputArray)
        return outputArray[0][0] > 0.5f
    }

    private fun convertMfccToByteBuffer(mfccArray: Array<FloatArray>, inputShape: IntArray): ByteBuffer {
        val (batchSize, timeSteps, features) = inputShape
        val byteBuffer = ByteBuffer.allocateDirect(batchSize * timeSteps * features * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until timeSteps) {
            val frame = if (i < mfccArray.size) mfccArray[i] else FloatArray(features)
            for (j in 0 until features) {
                byteBuffer.putFloat(if (j < frame.size) frame[j] else 0f)
            }
        }
        return byteBuffer
    }

    private suspend fun loadAndInitializeInterpreter() {
        withContext(Dispatchers.IO) {
            try {
                val modelFile = loadModelFile("BiLSTM.tflite")
                if (modelFile != null) {
                    val options = Interpreter.Options().apply {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                    }
                    interpreter = Interpreter(modelFile, options)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "모델 로드 완료", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "모델 파일이 없습니다.", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("Model Load Error", "모델 로드 중 오류 발생: ${e.message}")
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "모델 로드 실패", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer? {
        return try {
            assets.openFd(modelName).use { fileDescriptor ->
                FileInputStream(fileDescriptor.fileDescriptor).channel.use { fileChannel ->
                    val startOffset = fileDescriptor.startOffset
                    val declaredLength = fileDescriptor.declaredLength
                    fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
                }
            }
        } catch (e: Exception) {
            Log.e("Model Load Error", "Error loading model file", e)
            null
        }
    }

    private suspend fun extractMfccData(uri: Uri): Array<FloatArray>? {
        return withContext(Dispatchers.IO) {
            try {
                val inputStream = contentResolver.openInputStream(uri) ?: return@withContext null
                val audioData = readAudioData(inputStream)
                inputStream.close()

                if (audioData != null) {
                    val mfcc = MFCC()
                    val mfccFlatArray = mfcc.computeMFCC(audioData)
                    val numFrames = mfccFlatArray.size / MFCC.MEL_BANDS
                    Array(numFrames) { FloatArray(MFCC.MEL_BANDS) }.apply {
                        for (i in 0 until numFrames) {
                            System.arraycopy(mfccFlatArray, i * MFCC.MEL_BANDS, this[i], 0, MFCC.MEL_BANDS)
                        }
                    }
                } else null
            } catch (e: Exception) {
                Log.e("MFCC Extraction Error", "Error extracting MFCC: ", e)
                null
            }
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
            FloatArray(audioBytes.size / 2) { i ->
                ((audioBytes[i * 2].toInt() and 0xFF) or (audioBytes[i * 2 + 1].toInt() shl 8)) / 32768.0f
            }
        } catch (e: Exception) {
            Log.e("Audio Data Read Error", "Error reading audio data", e)
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close()
        interpreter = null
    }

    companion object {
        private const val PICK_AUDIO_REQUEST = 1
    }
}
