package com.example.stereowave

import android.content.Context
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class SimpleCameraManager(private val context: Context) {

    private var imageCapture: ImageCapture? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService

    private lateinit var sessionDir: File
    private lateinit var dataManager: SimpleDataManager
    private var absoluteStartTime: Long = 0
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-SSS", Locale.US)

    // Debug tracking
    private var isInitialized = false
    private var captureCount = 0

    fun initialize(sessionDirectory: File, dataManagerRef: SimpleDataManager, startTime: Long) {
        Log.d("SimpleCameraManager", "=== CAMERAX INITIALIZATION STARTED ===")
        sessionDir = sessionDirectory
        dataManager = dataManagerRef
        absoluteStartTime = startTime

        Log.d("SimpleCameraManager", "Session directory: ${sessionDir.absolutePath}")

        // Create camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize CameraX
        initializeCameraX()
    }

    private fun initializeCameraX() {
        Log.d("SimpleCameraManager", "Initializing CameraX...")

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                Log.d("SimpleCameraManager", "CameraX provider ready")
                cameraProvider = cameraProviderFuture.get()
                startCamera()
            } catch (e: Exception) {
                Log.e("SimpleCameraManager", "CameraX initialization failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun startCamera() {
        try {
            Log.d("SimpleCameraManager", "=== STARTING CAMERAX CAMERA ===")

            val cameraProvider = this.cameraProvider ?: return

            // Create ImageCapture use case with high quality settings
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(95)
                .setTargetRotation(0) // No rotation for marine applications
                .build()

            Log.d("SimpleCameraManager", "ImageCapture use case created")

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind all use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera - just ImageCapture, no preview needed for our use case
                camera = cameraProvider.bindToLifecycle(
                    context as LifecycleOwner,
                    cameraSelector,
                    imageCapture
                )

                Log.d("SimpleCameraManager", "=== CAMERAX CAMERA STARTED SUCCESSFULLY ===")
                isInitialized = true

            } catch (e: Exception) {
                Log.e("SimpleCameraManager", "Camera binding failed", e)
            }

        } catch (e: Exception) {
            Log.e("SimpleCameraManager", "Failed to start camera", e)
        }
    }

    fun captureImage(
        actualTimestamp: Long,
        sequenceNumber: Int,
        expectedTimestamp: Long = actualTimestamp,
        timingError: Long = 0
    ) {
        Log.d("SimpleCameraManager", "=== CAMERAX CAPTURE REQUEST #$sequenceNumber ===")

        if (!isInitialized) {
            Log.e("SimpleCameraManager", "CameraX not initialized! Skipping capture.")
            return
        }

        val imageCapture = this.imageCapture ?: run {
            Log.e("SimpleCameraManager", "ImageCapture is null! Skipping capture.")
            return
        }

        // Create output file
        val timestamp = Date()
        val filename = "IMG_${dateFormat.format(timestamp)}.jpg"
        val outputFile = File(sessionDir, filename)

        // Create output file options
        val outputFileOptions = ImageCapture.OutputFileOptions.Builder(outputFile)
            .build()

        Log.d("SimpleCameraManager", "Taking photo: $filename")

        // Take picture
        imageCapture.takePicture(
            outputFileOptions,
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    captureCount++

                    val fileExists = outputFile.exists()
                    val fileSize = if (fileExists) outputFile.length() else 0

                    Log.d("SimpleCameraManager", "=== CAMERAX IMAGE SAVED SUCCESSFULLY ===")
                    Log.d("SimpleCameraManager", "Filename: $filename")
                    Log.d("SimpleCameraManager", "File size: $fileSize bytes")
                    Log.d("SimpleCameraManager", "Total captures: $captureCount")
                    Log.d("SimpleCameraManager", "File path: ${outputFile.absolutePath}")

                    // Log timing data
                    dataManager.logImageTiming(
                        sessionDir,
                        sequenceNumber,
                        expectedTimestamp,
                        actualTimestamp,
                        absoluteStartTime
                    )
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("SimpleCameraManager", "=== CAMERAX CAPTURE FAILED ===")
                    Log.e("SimpleCameraManager", "Error code: ${exception.imageCaptureError}")
                    Log.e("SimpleCameraManager", "Error message: ${exception.message}")
                    Log.e("SimpleCameraManager", "Cause: ${exception.cause}")
                }
            }
        )
    }

    fun cleanup() {
        Log.d("SimpleCameraManager", "=== CAMERAX CLEANUP ===")
        Log.d("SimpleCameraManager", "Total images captured: $captureCount")

        try {
            cameraProvider?.unbindAll()
            cameraExecutor.shutdown()

            cameraProvider = null
            imageCapture = null
            camera = null
            isInitialized = false
            captureCount = 0

            Log.d("SimpleCameraManager", "CameraX cleanup completed")

        } catch (e: Exception) {
            Log.e("SimpleCameraManager", "Error during cleanup", e)
        }
    }
}

data class CaptureMetadata(
    val timestamp: Long,
    val sequenceNumber: Int,
    val expectedTimestamp: Long = timestamp,
    val timingError: Long = 0
)