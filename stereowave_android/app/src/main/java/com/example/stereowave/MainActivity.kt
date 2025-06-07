package com.example.stereowave

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.Manifest
import android.app.DatePickerDialog
import android.app.TimePickerDialog
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.io.FileOutputStream

class MainActivity : AppCompatActivity(), LifecycleOwner {

    // UI Components
    private lateinit var phoneIdEdit: EditText
    private lateinit var samplingRateSpinner: Spinner
    private lateinit var durationSpinner: Spinner
    private lateinit var startDateButton: Button
    private lateinit var startTimeButton: Button
    private lateinit var currentUtcText: TextView
    private lateinit var targetUtcText: TextView
    private lateinit var gpsCheckBox: CheckBox
    private lateinit var imuCheckBox: CheckBox
    private lateinit var statusText: TextView
    private lateinit var scheduleButton: Button
    private lateinit var stopButton: Button

    // Managers
    private lateinit var cameraManager: SimpleCameraManager
    private lateinit var dataManager: SimpleDataManager
    private lateinit var sensorManager: SimpleSensorManager
    private lateinit var syncManager: SimpleSyncManager

    // Wake lock management
    private lateinit var powerManager: PowerManager
    private var wakeLock: PowerManager.WakeLock? = null

    // Session state
    private var isCapturing = false
    private var captureJob: Job? = null
    private val handler = Handler(Looper.getMainLooper())
    private var utcTimeUpdateJob: Job? = null

    // Target UTC time
    private var targetUtcTimestamp: Long = 0
    private val utcTimeFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    private val utcDateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)
    private val utcClockFormat = SimpleDateFormat("HH:mm:ss", Locale.US)

    // Constants
    private val PERMISSIONS_REQUEST_CODE = 1001
    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.WRITE_EXTERNAL_STORAGE,
        Manifest.permission.READ_EXTERNAL_STORAGE
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Set all date formatters to UTC
        utcTimeFormat.timeZone = TimeZone.getTimeZone("UTC")
        utcDateFormat.timeZone = TimeZone.getTimeZone("UTC")
        utcClockFormat.timeZone = TimeZone.getTimeZone("UTC")

        // Initialize wake lock management
        initializeWakeLock()

        initializeViews()
        setupSpinners()
        setupClickListeners()
        initializeManagers()
        startUtcTimeUpdates()

        // Set default target time to 5 minutes from now
        setDefaultTargetTime()

        // Check permissions
        if (!hasAllPermissions()) {
            requestPermissions()
        } else {
            initializeCamera()
        }

        updateStatus("Ready to schedule capture")
    }

    private fun initializeWakeLock() {
        powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager

        // Keep screen on during the entire app lifecycle
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        Log.d("MainActivity", "Screen keep-on flag set for entire app session")
    }

    private fun acquireWakeLock() {
        try {
            // Create a wake lock to prevent the device from sleeping during capture
            wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK or PowerManager.ACQUIRE_CAUSES_WAKEUP,
                "StereoWave:CaptureSession"
            )

            // Acquire the wake lock
            wakeLock?.acquire(60 * 60 * 1000L) // Max 1 hour timeout as safety

            Log.d("MainActivity", "=== WAKE LOCK ACQUIRED ===")
            Log.d("MainActivity", "Device will stay awake during capture session")

        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to acquire wake lock", e)
            updateStatus("Warning: Could not prevent device sleep during capture")
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.let { lock ->
                if (lock.isHeld) {
                    lock.release()
                    Log.d("MainActivity", "=== WAKE LOCK RELEASED ===")
                }
            }
            wakeLock = null
        } catch (e: Exception) {
            Log.e("MainActivity", "Error releasing wake lock", e)
        }
    }

    private fun initializeViews() {
        phoneIdEdit = findViewById(R.id.phoneIdEdit)
        samplingRateSpinner = findViewById(R.id.samplingRateSpinner)
        durationSpinner = findViewById(R.id.durationSpinner)
        startDateButton = findViewById(R.id.startDateButton)
        startTimeButton = findViewById(R.id.startTimeButton)
        currentUtcText = findViewById(R.id.currentUtcText)
        targetUtcText = findViewById(R.id.targetUtcText)
        gpsCheckBox = findViewById(R.id.gpsCheckBox)
        imuCheckBox = findViewById(R.id.imuCheckBox)
        statusText = findViewById(R.id.statusText)
        scheduleButton = findViewById(R.id.scheduleButton)
        stopButton = findViewById(R.id.stopButton)

        // Set default phone ID
        phoneIdEdit.setText("LEFT")
        stopButton.isEnabled = false
    }

    private fun setupSpinners() {
        // Sampling rates
        val samplingRates = arrayOf("0.5 Hz", "1.0 Hz", "2.0 Hz", "3.0 Hz", "5.0 Hz")
        samplingRateSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, samplingRates)
        samplingRateSpinner.setSelection(2) // Default 2.0 Hz

        // Duration options
        val durations = arrayOf("5 min", "10 min", "15 min", "20 min", "30 min", "45 min", "60 min")
        durationSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, durations)
        durationSpinner.setSelection(3) // Default 20 min
    }

    private fun setupClickListeners() {
        scheduleButton.text = "SCHEDULE"
        scheduleButton.setOnClickListener {
            if (!isCapturing) {
                startScheduledCapture()
            }
        }

        stopButton.text = "STOP"
        stopButton.isEnabled = false
        stopButton.setOnClickListener {
            stopCapture()
        }

        startDateButton.setOnClickListener {
            showDatePicker()
        }

        startTimeButton.setOnClickListener {
            showTimePicker()
        }
    }

    private fun testStorageSaving() {
        updateStatus("Testing storage saving...")

        try {
            val testDir = File(dataManager.getStoragePath(), "storage_test_${System.currentTimeMillis()}")
            val dirCreated = testDir.mkdirs()

            updateStatus("Step 1: Created test directory\nPath: ${testDir.absolutePath}\nSuccess: $dirCreated")

            // Create a dummy file
            val testFile = File(testDir, "test_file.txt")
            val testData = "This is a test file - timestamp: ${System.currentTimeMillis()}"

            FileOutputStream(testFile).use { output ->
                output.write(testData.toByteArray())
                output.flush()
            }

            // Verify it was saved
            val fileExists = testFile.exists()
            val fileSize = if (fileExists) testFile.length() else 0

            var result = "=== STORAGE TEST COMPLETE ===\n"
            result += "Directory: ${testDir.name}\n"
            result += "File exists: $fileExists\n"
            result += "File size: $fileSize bytes\n\n"

            if (fileExists && fileSize > 0) {
                result += "✅ STORAGE WORKS!\n"
                result += "File system permissions are OK.\n"
                result += "Problem is in camera image capture.\n\n"
                result += "Now press TEST CAMERA button."
            } else {
                result += "❌ STORAGE FAILED!\n"
                result += "File system permission problem.\n\n"
                result += "Try:\n"
                result += "1. Settings > Apps > StereoWave > Permissions\n"
                result += "2. Enable all storage permissions"
            }

            updateStatus(result)

        } catch (e: Exception) {
            updateStatus("Storage test error: ${e.message}\n\nThis is definitely a storage permission issue.")
        }
    }

    private fun initializeManagers() {
        dataManager = SimpleDataManager(this)
        sensorManager = SimpleSensorManager(this)
        syncManager = SimpleSyncManager()
        cameraManager = SimpleCameraManager(this)
    }

    private fun startUtcTimeUpdates() {
        utcTimeUpdateJob = CoroutineScope(Dispatchers.Main).launch {
            while (true) {
                val currentUtc = System.currentTimeMillis()
                val utcString = utcClockFormat.format(Date(currentUtc))
                currentUtcText.text = "Current UTC: $utcString"

                // Update target time display
                if (targetUtcTimestamp > 0) {
                    val targetString = utcTimeFormat.format(Date(targetUtcTimestamp))
                    val remaining = targetUtcTimestamp - currentUtc
                    if (remaining > 0) {
                        val remainingMin = remaining / (60 * 1000)
                        val remainingSec = (remaining % (60 * 1000)) / 1000
                        targetUtcText.text = "Target: $targetString UTC\nIn: ${remainingMin}m ${remainingSec}s"
                    } else {
                        targetUtcText.text = "Target: $targetString UTC\nPAST"
                    }
                }

                delay(1000)
            }
        }
    }

    private fun setDefaultTargetTime() {
        // Set default to 5 minutes from now, rounded to next minute
        val currentTime = System.currentTimeMillis()
        val fiveMinutesLater = currentTime + (5 * 60 * 1000)

        // Round up to next minute boundary
        val calendar = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
        calendar.timeInMillis = fiveMinutesLater
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        calendar.add(Calendar.MINUTE, 1)

        targetUtcTimestamp = calendar.timeInMillis
        updateDateTimeButtons()
    }

    private fun showDatePicker() {
        val calendar = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
        if (targetUtcTimestamp > 0) {
            calendar.timeInMillis = targetUtcTimestamp
        }

        val datePickerDialog = DatePickerDialog(
            this,
            { _, year, month, dayOfMonth ->
                val newCalendar = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
                newCalendar.timeInMillis = targetUtcTimestamp
                newCalendar.set(Calendar.YEAR, year)
                newCalendar.set(Calendar.MONTH, month)
                newCalendar.set(Calendar.DAY_OF_MONTH, dayOfMonth)
                targetUtcTimestamp = newCalendar.timeInMillis
                updateDateTimeButtons()
            },
            calendar.get(Calendar.YEAR),
            calendar.get(Calendar.MONTH),
            calendar.get(Calendar.DAY_OF_MONTH)
        )

        datePickerDialog.show()
    }

    private fun showTimePicker() {
        val calendar = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
        if (targetUtcTimestamp > 0) {
            calendar.timeInMillis = targetUtcTimestamp
        }

        val timePickerDialog = TimePickerDialog(
            this,
            { _, hourOfDay, minute ->
                val newCalendar = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
                newCalendar.timeInMillis = targetUtcTimestamp
                newCalendar.set(Calendar.HOUR_OF_DAY, hourOfDay)
                newCalendar.set(Calendar.MINUTE, minute)
                newCalendar.set(Calendar.SECOND, 0)
                newCalendar.set(Calendar.MILLISECOND, 0)
                targetUtcTimestamp = newCalendar.timeInMillis
                updateDateTimeButtons()
            },
            calendar.get(Calendar.HOUR_OF_DAY),
            calendar.get(Calendar.MINUTE),
            true // 24-hour format
        )

        timePickerDialog.show()
    }

    private fun updateDateTimeButtons() {
        if (targetUtcTimestamp > 0) {
            val dateString = utcDateFormat.format(Date(targetUtcTimestamp))
            val timeString = utcClockFormat.format(Date(targetUtcTimestamp))
            startDateButton.text = "Date: $dateString"
            startTimeButton.text = "Time: $timeString UTC"
        }
    }

    private fun initializeCamera() {
        try {
            val storageInfo = dataManager.getStorageInfo()
            updateStatus("Camera ready. Set UTC target time and press SCHEDULE.\n\n" +
                    "📁 Storage: ${storageInfo.type}\n" +
                    "📍 Location: ${storageInfo.accessInstructions}")
        } catch (e: Exception) {
            updateStatus("Camera initialization failed: ${e.message}")
            Log.e("MainActivity", "Camera init failed", e)
        }
    }

    private fun startScheduledCapture() {
        if (isCapturing) return

        val phoneId = phoneIdEdit.text.toString().trim().uppercase()
        if (phoneId.isEmpty()) {
            updateStatus("Please enter phone ID (LEFT or RIGHT)")
            return
        }

        if (targetUtcTimestamp <= 0) {
            updateStatus("Please set target UTC date/time")
            return
        }

        val currentTime = System.currentTimeMillis()
        if (targetUtcTimestamp <= currentTime) {
            updateStatus("Target time must be in the future!")
            return
        }

        val samplingRate = getSamplingRate()
        val duration = getDuration()
        val sessionId = UUID.randomUUID().toString()

        // ACQUIRE WAKE LOCK BEFORE STARTING CAPTURE
        acquireWakeLock()

        isCapturing = true
        scheduleButton.isEnabled = false
        stopButton.isEnabled = true

        val targetString = utcTimeFormat.format(Date(targetUtcTimestamp))
        updateStatus("SYNC BOTH PHONES TO START AT:\n$targetString UTC\nPhone: $phoneId\nWaiting for target time...\n\n🔒 Wake lock acquired - device will stay awake")

        // Start countdown to absolute UTC time
        startUtcCountdown(targetUtcTimestamp) {
            startCaptureSession(sessionId, phoneId, samplingRate, duration, targetUtcTimestamp)
        }
    }

    private fun startUtcCountdown(absoluteUtcTime: Long, onComplete: () -> Unit) {
        val targetString = utcTimeFormat.format(Date(absoluteUtcTime))

        val countdownRunnable = object : Runnable {
            override fun run() {
                val currentTime = System.currentTimeMillis()
                val remainingMs = absoluteUtcTime - currentTime

                if (remainingMs > 5000 && isCapturing) {
                    val remainingMin = remainingMs / (60 * 1000)
                    val remainingSec = (remainingMs % (60 * 1000)) / 1000
                    val currentString = utcClockFormat.format(Date(currentTime))

                    updateStatus("CAPTURE STARTS AT:\n$targetString UTC\n" +
                            "Current UTC: $currentString\n" +
                            "Starting in: ${remainingMin}m ${remainingSec}s\n" +
                            "Get ready and stabilize phones!\n\n🔒 Device staying awake")

                    handler.postDelayed(this, 1000)
                } else if (remainingMs > 0 && isCapturing) {
                    // Final countdown with second precision
                    val remainingSec = (remainingMs + 999) / 1000
                    updateStatus("CAPTURE STARTS AT:\n$targetString UTC\n" +
                            "Starting in: ${remainingSec} seconds\n" +
                            "READY...\n\n🔒 Device staying awake")
                    handler.postDelayed(this, 500)
                } else if (isCapturing) {
                    // Start capture at exact UTC time
                    val actualStartTime = System.currentTimeMillis()
                    val timingError = actualStartTime - absoluteUtcTime
                    Log.d("MainActivity", "Capture started with ${timingError}ms timing error")
                    onComplete()
                }
            }
        }

        handler.post(countdownRunnable)
    }

    private fun startCaptureSession(
        sessionId: String,
        phoneId: String,
        samplingRate: Float,
        duration: Int,
        absoluteUtcTime: Long
    ) {
        try {
            val sessionDir = dataManager.startSession(
                sessionId, phoneId, samplingRate, duration,
                gpsCheckBox.isChecked, imuCheckBox.isChecked, absoluteUtcTime
            )

            cameraManager.initialize(sessionDir, dataManager, absoluteUtcTime)

            if (gpsCheckBox.isChecked) {
                sensorManager.startGPSLogging(sessionDir)
            }
            if (imuCheckBox.isChecked) {
                sensorManager.startIMULogging(sessionDir)
            }

            val actualStart = System.currentTimeMillis()
            val timingError = actualStart - absoluteUtcTime
            val targetString = utcTimeFormat.format(Date(absoluteUtcTime))
            val actualString = utcTimeFormat.format(Date(actualStart))

            updateStatus("CAPTURE STARTED!\n" +
                    "Target: $targetString UTC\n" +
                    "Actual: $actualString UTC\n" +
                    "Error: ${timingError}ms\n" +
                    "Phone: $phoneId, Rate: ${samplingRate}Hz\n\n🔒 Device staying awake during capture")

            captureJob = CoroutineScope(Dispatchers.Main).launch {
                runAbsoluteTimingCaptureLoop(samplingRate, duration, absoluteUtcTime)
            }

        } catch (e: Exception) {
            updateStatus("Failed to start capture: ${e.message}")
            Log.e("MainActivity", "Capture start failed", e)
            stopCapture()
        }
    }

    private suspend fun runAbsoluteTimingCaptureLoop(
        samplingRate: Float,
        durationMinutes: Int,
        absoluteStartTime: Long
    ) {
        val intervalMs = (1000.0f / samplingRate).toLong()
        val totalImages = (samplingRate * durationMinutes * 60).toInt()
        var imageCount = 0

        val endTime = absoluteStartTime + (durationMinutes * 60 * 1000)

        while (isCapturing && System.currentTimeMillis() < endTime && imageCount < totalImages) {
            // Calculate when this specific image should be captured
            val targetCaptureTime = absoluteStartTime + (imageCount * intervalMs)
            val currentTime = System.currentTimeMillis()

            // Wait until the exact target time
            val waitTime = targetCaptureTime - currentTime
            if (waitTime > 0) {
                delay(waitTime)
            }

            // Capture at the target time
            val actualCaptureTime = System.currentTimeMillis()
            val timingError = actualCaptureTime - targetCaptureTime

            cameraManager.captureImage(actualCaptureTime, imageCount + 1, targetCaptureTime, timingError)
            dataManager.incrementImageCount()
            imageCount++

            // Update status
            val elapsed = (actualCaptureTime - absoluteStartTime) / 1000
            val remaining = ((endTime - actualCaptureTime) / 1000).coerceAtLeast(0)
            val actualRate = if (elapsed > 0) imageCount / elapsed.toFloat() else 0f

            withContext(Dispatchers.Main) {
                updateStatus("Capturing... ${imageCount}/${totalImages}\n" +
                        "Elapsed: ${elapsed}s, Remaining: ${remaining}s\n" +
                        "Rate: ${"%.2f".format(actualRate)}Hz\n" +
                        "Last timing error: ${timingError}ms\n\n🔒 Device staying awake")
            }
        }

        withContext(Dispatchers.Main) {
            updateStatus("Capture completed! Images: $imageCount\nSaving final data...\n\n🔓 Releasing wake lock...")
            stopCapture()
        }
    }

    private fun stopCapture() {
        if (!isCapturing) return

        isCapturing = false
        captureJob?.cancel()

        // Stop all systems
        sensorManager.stopLogging()
        cameraManager.cleanup()

        // RELEASE WAKE LOCK WHEN CAPTURE ENDS
        releaseWakeLock()

        val sessionPath = dataManager.getSessionPath()
        val storageInfo = dataManager.getStorageInfo()
        dataManager.endSession()

        // Update UI
        scheduleButton.isEnabled = true
        stopButton.isEnabled = false

        updateStatus("Capture stopped.\n\n" +
                "📁 Data saved to PUBLIC storage:\n" +
                "${storageInfo.accessInstructions}\n\n" +
                "📍 Full path:\n$sessionPath\n\n" +
                "Ready for next session.\n\n🔓 Wake lock released - device can sleep")

        Log.d("MainActivity", "Capture session ended")
    }

    private fun getSamplingRate(): Float {
        val selected = samplingRateSpinner.selectedItem.toString()
        return selected.replace(" Hz", "").toFloat()
    }

    private fun getDuration(): Int {
        val selected = durationSpinner.selectedItem.toString()
        return selected.replace(" min", "").toInt()
    }

    private fun updateStatus(message: String) {
        runOnUiThread {
            statusText.text = message
            Log.d("MainActivity", "Status: $message")
        }
    }

    private fun hasAllPermissions(): Boolean {
        return requiredPermissions.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, requiredPermissions, PERMISSIONS_REQUEST_CODE)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                initializeCamera()
            } else {
                updateStatus("Permissions required for camera and storage access")
                Toast.makeText(this, "All permissions are required for the app to work", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        utcTimeUpdateJob?.cancel()
        if (isCapturing) {
            stopCapture()
        }
        // Release wake lock if still held
        releaseWakeLock()

        // Remove the keep screen on flag
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onPause() {
        super.onPause()
        // Keep capture running in background - wake lock will maintain the session
        Log.d("MainActivity", "App paused but capture session continues with wake lock")
    }

    override fun onResume() {
        super.onResume()
        Log.d("MainActivity", "App resumed - capture session status: $isCapturing")
    }

    private fun testCameraCapture() {
        updateStatus("Starting camera test...\nWait 15 seconds...")

        val testDir = File(dataManager.getStoragePath(), "camera_test_${System.currentTimeMillis()}")
        testDir.mkdirs()

        updateStatus("Initializing camera...\nTest directory: ${testDir.name}")

        cameraManager.initialize(testDir, dataManager, System.currentTimeMillis())

        // Wait 5 seconds for camera initialization
        handler.postDelayed({
            updateStatus("Taking test photo 1...")
            cameraManager.captureImage(System.currentTimeMillis(), 1)

            handler.postDelayed({
                updateStatus("Taking test photo 2...")
                cameraManager.captureImage(System.currentTimeMillis(), 2)

                handler.postDelayed({
                    updateStatus("Taking test photo 3...")
                    cameraManager.captureImage(System.currentTimeMillis(), 3)

                    handler.postDelayed({
                        // Check results
                        val imageFiles = testDir.listFiles { file -> file.name.endsWith(".jpg") }
                        val allFiles = testDir.listFiles()

                        var result = "=== CAMERA TEST COMPLETE ===\n"
                        result += "Directory: ${testDir.name}\n"
                        result += "Total files: ${allFiles?.size ?: 0}\n"
                        result += "Image files (.jpg): ${imageFiles?.size ?: 0}\n\n"

                        if ((imageFiles?.size ?: 0) > 0) {
                            result += "✅ SUCCESS! Camera working!\n\n"
                            imageFiles?.forEach { file ->
                                result += "📷 ${file.name}: ${file.length()} bytes\n"
                            }
                        } else {
                            result += "❌ NO IMAGES SAVED\n\n"
                            result += "Camera LED lit up but no files saved.\n"
                            result += "This means camera capture is failing\n"
                            result += "in the image processing pipeline.\n\n"

                            if ((allFiles?.size ?: 0) > 0) {
                                result += "Other files found:\n"
                                allFiles?.forEach { file ->
                                    result += "📄 ${file.name}: ${file.length()} bytes\n"
                                }
                            }
                        }

                        cameraManager.cleanup()
                        updateStatus(result)

                    }, 6000) // Wait 6 seconds for images to save
                }, 2000)
            }, 2000)
        }, 5000) // Wait 5 seconds for camera init
    }

    private fun checkCameraStatus() {
        try {
            val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = cameraManager.cameraIdList

            var statusMessage = "=== CAMERA STATUS CHECK ===\n"
            statusMessage += "Available cameras: ${cameraIds.size}\n"

            cameraIds.forEachIndexed { index, id ->
                val characteristics = cameraManager.getCameraCharacteristics(id)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                val facingStr = when (facing) {
                    CameraCharacteristics.LENS_FACING_BACK -> "Back"
                    CameraCharacteristics.LENS_FACING_FRONT -> "Front"
                    else -> "Other"
                }
                statusMessage += "Camera $index: ID=$id ($facingStr)\n"
            }

            // Check permissions
            val cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            val storagePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)

            statusMessage += "\nPermissions:\n"
            statusMessage += "Camera: ${if (cameraPermission == PackageManager.PERMISSION_GRANTED) "✅ Granted" else "❌ Denied"}\n"
            statusMessage += "Storage: ${if (storagePermission == PackageManager.PERMISSION_GRANTED) "✅ Granted" else "❌ Denied"}\n"

            updateStatus(statusMessage)

        } catch (e: Exception) {
            updateStatus("Camera status check failed: ${e.message}")
        }
    }
}