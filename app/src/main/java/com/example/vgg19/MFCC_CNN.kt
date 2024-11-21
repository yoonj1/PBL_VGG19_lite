package com.example.vgg19

import kotlin.math.*

class MFCC_CNN {
    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME_SIZE = 1024
        const val MEL_BANDS = 13
        const val FFT_SIZE = 2048
    }

    fun computeCnnMFCC(signal: FloatArray): Array<Array<FloatArray>> {
        if (signal.isEmpty()) throw IllegalArgumentException("입력 신호가 비어 있습니다.")
        val numFrames = signal.size / FRAME_SIZE
        if (numFrames == 0) throw IllegalArgumentException("프레임 개수가 충분하지 않습니다.")

        val mfccs = Array(numFrames) { FloatArray(MEL_BANDS) }
        for (i in 0 until numFrames) {
            val frame = signal.copyOfRange(i * FRAME_SIZE, (i + 1) * FRAME_SIZE)
            val spectrum = computeFFT(frame)
            val melSpectrum = applyMelFilterBank(spectrum)
            val logMelSpectrum = applyLog(melSpectrum)
            val mfcc = computeDCT(logMelSpectrum)
            mfccs[i] = mfcc
        }

        val resizedMfcc = adjustMFCCSize(mfccs, timeSteps = 224, features = 224)
        val cnnInput = Array(224) { Array(224) { FloatArray(3) } }
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val normalizedValue = resizedMfcc[i][j]
                cnnInput[i][j].fill(normalizedValue)
            }
        }
        return cnnInput
    }

    private fun computeFFT(frame: FloatArray): FloatArray {
        val paddedFrame = if (frame.size < FFT_SIZE) frame.copyOf(FFT_SIZE) else frame
        val fft = FloatArray(FFT_SIZE)
        System.arraycopy(paddedFrame, 0, fft, 0, paddedFrame.size)

        for (i in fft.indices) fft[i] *= 0.54f - 0.46f * cos(2 * Math.PI * i / fft.size).toFloat()

        val magnitude = FloatArray(FFT_SIZE / 2)
        for (i in magnitude.indices) {
            val real = fft[2 * i]
            val imag = fft[2 * i + 1]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        return magnitude
    }

    private fun applyMelFilterBank(spectrum: FloatArray): FloatArray {
        val numMelFilters = MEL_BANDS + 2
        val melFilters = FloatArray(numMelFilters)
        val melMax = hzToMel(SAMPLE_RATE / 2f)

        for (i in melFilters.indices) melFilters[i] = melToHz(melMax * i / (numMelFilters - 1))

        val melSpectrum = FloatArray(MEL_BANDS)
        for (m in 1..MEL_BANDS) {
            val leftFreq = melFilters[m - 1].toInt()
            val centerFreq = melFilters[m].toInt()
            val rightFreq = melFilters[m + 1].toInt()

            for (f in leftFreq until rightFreq) {
                if (f < spectrum.size) {
                    val weight = when {
                        f < centerFreq -> (f - leftFreq).toFloat() / (centerFreq - leftFreq)
                        else -> (rightFreq - f).toFloat() / (rightFreq - centerFreq)
                    }
                    melSpectrum[m - 1] += spectrum[f] * weight
                }
            }
        }
        return melSpectrum
    }

    private fun applyLog(melSpectrum: FloatArray): FloatArray {
        return melSpectrum.map { log10(1 + it) }.toFloatArray()
    }

    private fun computeDCT(logMelSpectrum: FloatArray): FloatArray {
        val mfcc = FloatArray(MEL_BANDS)
        for (k in 0 until MEL_BANDS) {
            for (n in 0 until MEL_BANDS) {
                mfcc[k] += logMelSpectrum[n] * cos(Math.PI * k * (n + 0.5) / MEL_BANDS).toFloat()
            }
            mfcc[k] *= sqrt(2.0f / MEL_BANDS)
        }
        return mfcc
    }

    private fun hzToMel(hz: Float): Float = (2595 * log10(1 + hz / 700)).toFloat()

    private fun melToHz(mel: Float): Float = (700 * (10.0.pow(mel / 2595.0) - 1)).toFloat()

    private fun adjustMFCCSize(mfccs: Array<FloatArray>, timeSteps: Int, features: Int): Array<FloatArray> {
        val adjustedMFCC = Array(timeSteps) { FloatArray(features) }
        for (i in 0 until timeSteps) {
            val frame = mfccs.getOrNull(i) ?: FloatArray(features)
            for (j in 0 until features) adjustedMFCC[i][j] = frame.getOrElse(j) { 0f }
        }
        return adjustedMFCC
    }
}
