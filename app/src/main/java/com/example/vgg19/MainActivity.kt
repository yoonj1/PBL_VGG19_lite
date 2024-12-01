package com.example.vgg19

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private var audioUri: Uri? = null
    private var mfccArray: Array<Array<FloatArray>>? = null
    private var interpreter: Interpreter? = null
    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var isRecording = false

    // UI 요소
    private lateinit var resultTextView: TextView
    private lateinit var recordButton: ImageView
    private lateinit var selectAudioButton: ImageView
    private lateinit var extractMfccButton: Button
    private lateinit var classifyButton: Button

    companion object {
        private const val PICK_AUDIO_REQUEST = 1
        private const val RECORD_AUDIO_PERMISSION_REQUEST = 2
    }

    private val mfccCnn = MFCC_CNN()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        checkPermissions()

        // UI 초기화
        resultTextView = findViewById(R.id.tv_result)
        recordButton = findViewById(R.id.btn_record)
        selectAudioButton = findViewById(R.id.btn_upload)
        extractMfccButton = findViewById(R.id.btn_mfcc)
        classifyButton = findViewById(R.id.btn_result)

        // 버튼 리스너 설정
        recordButton.setOnClickListener { toggleRecording() }
        selectAudioButton.setOnClickListener { openAudioFilePicker() }
        extractMfccButton.setOnClickListener { extractMfcc() }
        classifyButton.setOnClickListener { classifyAudio() }

        // 모델 로드
        loadInterpreterForCNNModel()
    }

    private fun toggleRecording() {
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                RECORD_AUDIO_PERMISSION_REQUEST
            )
            return
        }

        val outputDir = cacheDir
        audioFile = File.createTempFile("recording", ".wav", outputDir)

        mediaRecorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
            setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
            setOutputFile(audioFile?.absolutePath)
            prepare()
            start()
        }

        isRecording = true
        recordButton.setImageResource(R.drawable.ic_stop) // 정지 버튼으로 아이콘 변경
        resultTextView.text = "녹음 중..."
    }

    private fun stopRecording() {
        mediaRecorder?.apply {
            stop()
            release()
        }
        mediaRecorder = null
        audioUri = Uri.fromFile(audioFile)

        isRecording = false
        recordButton.setImageResource(R.drawable.recode) // 녹음 버튼 아이콘으로 복원
        resultTextView.text = "녹음 완료"
    }

    private fun openAudioFilePicker() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply { type = "audio/*" }
        startActivityForResult(intent, PICK_AUDIO_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_AUDIO_REQUEST && resultCode == RESULT_OK && data != null) {
            audioUri = data.data
            resultTextView.text = "음성 파일 선택 완료"
        }
    }

    private fun extractMfcc() {
        if (audioUri == null) {
            Toast.makeText(this, "음성 파일을 먼저 선택하세요.", Toast.LENGTH_SHORT).show()
            return
        }

        CoroutineScope(Dispatchers.IO).launch {
            val mfccResult = try {
                val audioData = loadAudioData(audioUri!!) ?: FloatArray(0)
                mfccCnn.computeCnnMFCC(audioData)
            } catch (e: Exception) {
                Log.e("MFCC Error", "MFCC 추출 실패: ${e.message}")
                null
            }

            withContext(Dispatchers.Main) {
                if (mfccResult != null) {
                    mfccArray = mfccResult
                    resultTextView.text = "MFCC 추출 완료"
                } else {
                    resultTextView.text = "MFCC 추출 실패"
                }
            }
        }
    }

    private fun classifyAudio() {
        if (mfccArray == null) {
            Toast.makeText(this, "MFCC 데이터를 먼저 추출하세요.", Toast.LENGTH_SHORT).show()
            return
        }

        resultTextView.text = "분류 진행 중..."
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = runModelInference(mfccArray!!)
                val realAudioProbability = result
                val classificationResult = if (realAudioProbability >= 0.001f / 100) {
                    "진짜 음성"
                } else {
                    "가짜 음성"
                }

                withContext(Dispatchers.Main) {
                    resultTextView.text = "판별 결과: $classificationResult"
                }
            } catch (e: Exception) {
                Log.e("ClassificationError", "분류 실패: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "분류 실패: ${e.message}"
                }
            }
        }
    }


    private fun runModelInference(input: Array<Array<FloatArray>>): Float {
        val inputBuffer = convertToByteBuffer(input)
        val outputArray = Array(1) { FloatArray(1) }
        interpreter?.run(inputBuffer, outputArray)
        return outputArray[0][0]
    }

    private fun convertToByteBuffer(input: Array<Array<FloatArray>>): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder())
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                for (k in 0 until 3) {
                    buffer.putFloat(input[i][j][k])
                }
            }
        }
        return buffer
    }

    private fun loadInterpreterForCNNModel() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val modelBuffer = loadModelFile("vgg19_alternative_mode.tflite")
                interpreter = Interpreter(modelBuffer)
            } catch (e: Exception) {
                Log.e("ModelLoadError", "모델 로드 실패: ${e.message}")
            }
        }
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val assetFileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private suspend fun loadAudioData(uri: Uri): FloatArray? {
        return withContext(Dispatchers.IO) {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val byteArray = inputStream?.readBytes() ?: return@withContext null
                inputStream.close()

                val floatArray = FloatArray(byteArray.size / 2)
                for (i in floatArray.indices) {
                    val sample = (byteArray[i * 2].toInt() and 0xFF) or (byteArray[i * 2 + 1].toInt() shl 8)
                    floatArray[i] = sample / 32768.0f
                }
                floatArray
            } catch (e: Exception) {
                Log.e("AudioDataError", "오디오 데이터 로드 실패: ${e.message}")
                null
            }
        }
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_PERMISSION_REQUEST)
        }
    }
}
