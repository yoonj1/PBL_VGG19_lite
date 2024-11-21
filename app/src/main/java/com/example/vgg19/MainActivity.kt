package com.example.vgg19

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private var audioUri: Uri? = null
    private var mfccArray: Array<Array<FloatArray>>? = null
    private var interpreter: Interpreter? = null
    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null
    private lateinit var progressBar: ProgressBar
    private lateinit var resultTextView: TextView

    companion object {
        private const val PICK_AUDIO_REQUEST = 1
        private const val RECORD_AUDIO_PERMISSION_REQUEST = 2
    }

    private val mfccCnn = MFCC_CNN()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        checkPermissions() // 권한 확인
        progressBar = findViewById(R.id.progressBar)
        resultTextView = findViewById(R.id.resultTextView)

        findViewById<Button>(R.id.select_audio_button).setOnClickListener { openAudioFilePicker() }
        findViewById<Button>(R.id.start_recording_button).setOnClickListener { startRecording() }
        findViewById<Button>(R.id.stop_recording_button).setOnClickListener { stopRecording() }
        findViewById<Button>(R.id.extract_mfcc_button).setOnClickListener { extractMfcc() }
        findViewById<Button>(R.id.classify_button).setOnClickListener { classifyAudio() }

        loadInterpreterForCNNModel()
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

    private fun classifyAudio() {
        if (mfccArray == null) {
            Toast.makeText(this, "MFCC 데이터를 먼저 추출하세요.", Toast.LENGTH_SHORT).show()
            return
        }

        progressBar.visibility = View.VISIBLE
        resultTextView.text = "분류 진행 중..."

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = runModelInference(mfccArray as Array<Array<FloatArray>>)
                val realAudioProbability = result
                val fakeAudioProbability = 1f - result

                // 임계값 설정: 0.001%로 설정
                val threshold = 0.001f  // 0.001%

                // 판별: realAudioProbability가 0.001% 이상이면 "진짜 음성", 아니면 "가짜 음성"
                val classificationResult = if (realAudioProbability >= threshold / 100) {
                    "진짜 음성"
                } else {
                    "가짜 음성"
                }

                // 로그로 출력하여 결과 확인
                Log.d("Classification", "임계값: $threshold%, 진짜일 확률: ${realAudioProbability * 100}%, 판별 결과: $classificationResult")

                // 최종 판별 결과를 화면에 표시
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    resultTextView.text = """
                    진짜일 확률: ${String.format("%.8f", realAudioProbability * 100)}%
                    가짜일 확률: ${String.format("%.8f", fakeAudioProbability * 100)}%
                    판별 결과: $classificationResult
                """.trimIndent()
                }
            } catch (e: Exception) {
                Log.e("ClassificationError", "분류 실패: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    resultTextView.text = "분류 실패: ${e.message}"
                }
            }
        }
    }






    private fun runModelInference(input: Array<Array<FloatArray>>): Float {
        try {
            val inputBuffer = convertToByteBuffer(input)
            Log.d("ModelInference", "Input Buffer Size: ${inputBuffer.capacity()} bytes")

            val outputArray = Array(1) { FloatArray(1) }
            interpreter?.run(inputBuffer, outputArray)
            Log.d("ModelInference", "Output: ${outputArray[0][0]}")

            return outputArray[0][0]
        } catch (e: Exception) {
            Log.e("ModelInferenceError", "예측 실패: ${e.message}")
            throw e
        }
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

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
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

        resultTextView.text = "녹음 중..."
        findViewById<Button>(R.id.start_recording_button).isEnabled = false
        findViewById<Button>(R.id.stop_recording_button).isEnabled = true
    }

    private fun stopRecording() {
        mediaRecorder?.apply {
            stop()
            release()
        }
        mediaRecorder = null
        audioUri = Uri.fromFile(audioFile)

        resultTextView.text = "녹음 완료"
        findViewById<Button>(R.id.start_recording_button).isEnabled = true
        findViewById<Button>(R.id.stop_recording_button).isEnabled = false
    }

    private fun loadInterpreterForCNNModel() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val modelBuffer = loadModelFile("vgg19_alternative_mode.tflite")
                interpreter = Interpreter(modelBuffer)
                Log.d("ModelLoad", "모델 로드 성공")
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

    private fun extractMfcc() {
        if (audioUri == null) {
            Toast.makeText(this, "음성 파일을 먼저 선택하거나 녹음하세요.", Toast.LENGTH_SHORT).show()
            return
        }

        progressBar.visibility = View.VISIBLE

        CoroutineScope(Dispatchers.IO).launch {
            val mfccResult = try {
                val audioData = loadAudioData(audioUri!!)
                mfccCnn.computeCnnMFCC(audioData)
            } catch (e: Exception) {
                Log.e("MFCC Error", "MFCC 추출 실패: ${e.message}")
                null
            }

            withContext(Dispatchers.Main) {
                progressBar.visibility = View.GONE
                if (mfccResult != null) {
                    mfccArray = mfccResult
                    resultTextView.text = "MFCC 추출 완료"
                } else {
                    resultTextView.text = "MFCC 추출 실패"
                }
            }
        }
    }

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE
    )

    private val PERMISSION_REQUEST_CODE = 100

    private fun checkPermissions() {
        val missingPermissions = REQUIRED_PERMISSIONS.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.any { it != PackageManager.PERMISSION_GRANTED }) {
                Toast.makeText(this, "권한이 필요합니다!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loadAudioData(uri: Uri): FloatArray {
        try {
            val inputStream = contentResolver.openInputStream(uri)
                ?: throw IllegalArgumentException("오디오 파일을 열 수 없습니다.")
            val byteArray = inputStream.readBytes()
            inputStream.close()

            val floatArray = FloatArray(byteArray.size / 2)
            for (i in floatArray.indices) {
                val sample = (byteArray[i * 2].toInt() and 0xFF) or (byteArray[i * 2 + 1].toInt() shl 8)
                floatArray[i] = sample / 32768.0f
            }

            if (floatArray.isEmpty()) throw IllegalStateException("오디오 데이터가 비어 있습니다.")
            return floatArray
        } catch (e: Exception) {
            Log.e("AudioDataError", "오디오 데이터 로드 실패: ${e.message}")
            return FloatArray(0)
        }
    }
}
