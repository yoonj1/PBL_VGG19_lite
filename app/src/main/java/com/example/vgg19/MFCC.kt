package com.example.vgg19

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

class MFCC {
    companion object {
        const val SAMPLE_RATE = 44100
        const val FRAME_SIZE = 1024
        const val MEL_BANDS = 13
        const val FFT_SIZE = 2048
    }

    fun computeMFCC(signal: FloatArray): FloatArray {
        val numFrames = signal.size / FRAME_SIZE
        val mfccs = FloatArray(numFrames * MEL_BANDS)

        for (i in 0 until numFrames) {
            val frame = signal.copyOfRange(i * FRAME_SIZE, (i + 1) * FRAME_SIZE)
            val spectrum = computeFFT(frame)
            val melSpectrum = applyMelFilterBank(spectrum)
            val logMelSpectrum = applyLog(melSpectrum)
            val mfcc = computeDCT(logMelSpectrum)

            System.arraycopy(mfcc, 0, mfccs, i * MEL_BANDS, MEL_BANDS)
        }
        return mfccs
    }

    private fun computeFFT(frame: FloatArray): FloatArray {
        val paddedFrame = if (frame.size < FFT_SIZE) frame.copyOf(FFT_SIZE) else frame

        val fft = FloatFFT_1D(FFT_SIZE.toLong())
        fft.realForward(paddedFrame)

        val magnitude = FloatArray(FFT_SIZE / 2)
        for (i in magnitude.indices) {
            val real = paddedFrame[2 * i]
            val imag = paddedFrame[2 * i + 1]
            magnitude[i] = sqrt(real * real + imag * imag)
        }
        return magnitude
    }

    private fun applyMelFilterBank(spectrum: FloatArray): FloatArray {
        val numMelFilters = MEL_BANDS + 2
        val melFilters = FloatArray(numMelFilters)
        val melMax = hzToMel(SAMPLE_RATE / 2f)

        for (i in melFilters.indices) {
            melFilters[i] = melToHz(melMax * i / (numMelFilters - 1))
        }

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
        val logMelSpectrum = FloatArray(melSpectrum.size)
        for (i in melSpectrum.indices) {
            logMelSpectrum[i] = log10(1 + melSpectrum[i])  // 로그 변환
        }
        return logMelSpectrum
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

    private fun hzToMel(hz: Float): Float {
        return (2595 * log10(1 + hz / 700)).toFloat()
    }

    private fun melToHz(mel: Float): Float {
        return (700 * (10.0.pow(mel / 2595.0) - 1)).toFloat()
    }
}
