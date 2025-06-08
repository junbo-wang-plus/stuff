package com.example.stereowave

import android.content.Context
import android.os.Build
import android.os.Environment
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class SimpleDataManager(private val context: Context) {

    private var sessionConfig: SessionConfig? = null
    private var sessionDir: File? = null
    private var sessionStartTime: Long = 0
    private var imageCount: Int = 0

    fun startSession(
        sessionId: String,
        phoneId: String,
        samplingRate: Float,
        duration: Int,
        logGPS: Boolean,
        logIMU: Boolean,
        absoluteStartTime: Long
    ): File {
        sessionStartTime = absoluteStartTime

        // Create session directory with absolute start time
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date(absoluteStartTime))
        val sessionDirName = "session_${timestamp}_${phoneId}"
        sessionDir = File(getPublicStorageDirectory(), sessionDirName)

        val dirCreated = sessionDir!!.mkdirs()
        Log.d("SimpleDataManager", "Creating session directory: ${sessionDir!!.absolutePath}")
        Log.d("SimpleDataManager", "Directory creation success: $dirCreated")

        // Store session configuration
        sessionConfig = SessionConfig(
            sessionId = sessionId,
            phoneId = phoneId,
            startTime = absoluteStartTime,
            samplingRate = samplingRate,
            duration = duration,
            logGPS = logGPS,
            logIMU = logIMU
        )

        // Create initial session info file
        saveSessionInfo()
        createTimingLog(sessionDir!!, absoluteStartTime)

        Log.d("SimpleDataManager", "Started session: $sessionDirName")
        Log.d("SimpleDataManager", "Absolute start time: ${Date(absoluteStartTime)}")
        Log.d("SimpleDataManager", "PUBLIC storage path: ${sessionDir!!.absolutePath}")

        return sessionDir!!
    }

    private fun getPublicStorageDirectory(): File {
        // Use Documents/StereoWave for session data - visible in file managers
        val publicDocuments = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val stereoWaveDir = File(publicDocuments, "StereoWave")

        if (!stereoWaveDir.exists()) {
            val created = stereoWaveDir.mkdirs()
            Log.d("SimpleDataManager", "Created PUBLIC StereoWave directory: ${stereoWaveDir.absolutePath}")
            Log.d("SimpleDataManager", "Directory creation success: $created")

            if (!created) {
                Log.e("SimpleDataManager", "Failed to create public directory, falling back to app-specific storage")
                // Fallback to app-specific directory if public creation fails
                return getAppSpecificDirectory()
            }
        }

        Log.d("SimpleDataManager", "Using PUBLIC storage: ${stereoWaveDir.absolutePath}")
        return stereoWaveDir
    }

    private fun getAppSpecificDirectory(): File {
        // Fallback to app-specific external files directory
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        val storageDir = File(baseDir, "StereoWave")

        if (!storageDir.exists()) {
            storageDir.mkdirs()
            Log.d("SimpleDataManager", "Fallback: Created app-specific directory: ${storageDir.absolutePath}")
        }

        return storageDir
    }

    private fun saveSessionInfo() {
        val config = sessionConfig ?: return
        val sessionFile = File(sessionDir, "session_info.json")

        try {
            val currentTime = System.currentTimeMillis()
            val json = JSONObject().apply {
                put("sessionId", config.sessionId)
                put("phoneId", config.phoneId)
                put("absoluteStartTime", config.startTime)
                put("currentTime", currentTime)
                put("samplingRate", config.samplingRate)
                put("duration", config.duration)
                put("logGPS", config.logGPS)
                put("logIMU", config.logIMU)
                put("phoneModel", "${Build.MANUFACTURER} ${Build.MODEL}")
                put("androidVersion", Build.VERSION.RELEASE)
                put("appVersion", "1.0.0")
                put("imageCount", imageCount)
                put("sessionPath", sessionDir!!.absolutePath)
                put("isPublicStorage", isUsingPublicStorage())
                put("storageType", getStorageTypeDescription())
                put("timezoneOffset", TimeZone.getDefault().rawOffset)
                put("isDaylightSaving", TimeZone.getDefault().inDaylightTime(Date()))
            }

            FileWriter(sessionFile).use { writer ->
                writer.write(json.toString(2))
            }

            Log.d("SimpleDataManager", "Saved session info to PUBLIC location")

        } catch (e: Exception) {
            Log.e("SimpleDataManager", "Failed to save session info", e)
        }
    }

    private fun isUsingPublicStorage(): Boolean {
        val sessionPath = sessionDir?.absolutePath ?: ""
        return sessionPath.contains("/Documents/StereoWave") ||
                sessionPath.contains("/Pictures/StereoWave") ||
                sessionPath.contains("/DCIM/StereoWave")
    }

    private fun getStorageTypeDescription(): String {
        val sessionPath = sessionDir?.absolutePath ?: ""
        return when {
            sessionPath.contains("/Documents/StereoWave") -> "Public Documents Folder"
            sessionPath.contains("/Pictures/StereoWave") -> "Public Pictures Folder"
            sessionPath.contains("/DCIM/StereoWave") -> "Public DCIM Folder"
            sessionPath.contains("Android/data") -> "App-Specific Storage (Hidden)"
            else -> "Unknown Storage Location"
        }
    }

    fun createTimingLog(sessionDir: File, absoluteStartTime: Long) {
        val timingFile = File(sessionDir, "timing_log.csv")

        try {
            FileWriter(timingFile).use { writer ->
                writer.write("image_sequence,target_timestamp,actual_timestamp,timing_error_ms,cumulative_drift_ms\n")
            }
            Log.d("SimpleDataManager", "Created timing log file in PUBLIC storage")
        } catch (e: Exception) {
            Log.e("SimpleDataManager", "Failed to create timing log", e)
        }
    }

    fun logImageTiming(
        sessionDir: File,
        sequenceNumber: Int,
        targetTimestamp: Long,
        actualTimestamp: Long,
        absoluteStartTime: Long
    ) {
        val timingFile = File(sessionDir, "timing_log.csv")

        try {
            if (timingFile.exists()) {
                val timingError = actualTimestamp - targetTimestamp
                val cumulativeDrift = actualTimestamp - (absoluteStartTime + (sequenceNumber - 1) * getImageInterval())

                FileWriter(timingFile, true).use { writer ->
                    writer.write("$sequenceNumber,$targetTimestamp,$actualTimestamp,$timingError,$cumulativeDrift\n")
                }
            }
        } catch (e: Exception) {
            Log.e("SimpleDataManager", "Failed to log timing data", e)
        }
    }

    private fun getImageInterval(): Long {
        val config = sessionConfig
        return if (config != null) {
            (1000.0f / config.samplingRate).toLong()
        } else {
            500L // Default fallback
        }
    }

    fun incrementImageCount() {
        imageCount++

        // Update session info periodically (every 10 images)
        if (imageCount % 10 == 0) {
            saveSessionInfo()
        }
    }

    fun endSession() {
        val endTime = System.currentTimeMillis()
        val durationSeconds = (endTime - sessionStartTime) / 1000.0f
        val actualSamplingRate = if (durationSeconds > 0) imageCount / durationSeconds else 0.0f

        val config = sessionConfig
        if (config != null && sessionDir != null) {
            val finalInfoFile = File(sessionDir, "session_summary.json")

            try {
                val json = JSONObject().apply {
                    put("sessionId", config.sessionId)
                    put("phoneId", config.phoneId)
                    put("absoluteStartTime", config.startTime)
                    put("endTime", endTime)
                    put("targetSamplingRate", config.samplingRate)
                    put("actualSamplingRate", actualSamplingRate)
                    put("targetDuration", config.duration)
                    put("actualDuration", durationSeconds)
                    put("totalImages", imageCount)
                    put("logGPS", config.logGPS)
                    put("logIMU", config.logIMU)
                    put("phoneModel", "${Build.MANUFACTURER} ${Build.MODEL}")
                    put("androidVersion", Build.VERSION.RELEASE)
                    put("sessionPath", sessionDir!!.absolutePath)
                    put("isPublicStorage", isUsingPublicStorage())
                    put("storageType", getStorageTypeDescription())
                    put("fileManagerInstructions", getFileManagerInstructions())
                }

                FileWriter(finalInfoFile).use { writer ->
                    writer.write(json.toString(2))
                }

                Log.d("SimpleDataManager", "Session ended - Duration: ${durationSeconds}s, Images: $imageCount, Rate: ${"%.2f".format(actualSamplingRate)}Hz")
                Log.d("SimpleDataManager", "Final data saved to PUBLIC location: ${sessionDir!!.absolutePath}")

            } catch (e: Exception) {
                Log.e("SimpleDataManager", "Failed to save final session info", e)
            }
        }

        // Reset state
        sessionDir = null
        imageCount = 0
        sessionConfig = null
    }

    private fun getFileManagerInstructions(): String {
        return if (isUsingPublicStorage()) {
            "Files are saved in PUBLIC storage. Open your file manager app and navigate to Documents > StereoWave to find your session data."
        } else {
            "Files are in app-specific storage. Use Android Debug Bridge (ADB) or specialized apps to access Android/data/com.example.stereowave/files/"
        }
    }

    fun getSessionPath(): String? {
        return sessionDir?.absolutePath
    }

    fun listSessions(): List<File> {
        val storageDir = getPublicStorageDirectory()
        return storageDir.listFiles { file ->
            file.isDirectory && file.name.startsWith("session_")
        }?.sortedByDescending { it.lastModified() } ?: emptyList()
    }

    fun getSessionInfo(): SessionConfig? = sessionConfig

    fun getStoragePath(): String = getPublicStorageDirectory().absolutePath

    /**
     * Get human-readable storage information for UI display
     */
    fun getStorageInfo(): StorageInfo {
        val storageDir = getPublicStorageDirectory()
        val isPublic = storageDir.absolutePath.contains("/Documents/") ||
                storageDir.absolutePath.contains("/Pictures/") ||
                storageDir.absolutePath.contains("/DCIM/")

        return StorageInfo(
            path = storageDir.absolutePath,
            isPublic = isPublic,
            accessInstructions = if (isPublic) {
                "📁 Open file manager → Documents → StereoWave"
            } else {
                "⚠️ Hidden location - use ADB or specialized apps"
            },
            type = if (isPublic) "Public Documents" else "App-Specific"
        )
    }
}

data class SessionConfig(
    val sessionId: String,
    val phoneId: String,
    val startTime: Long,
    val samplingRate: Float,
    val duration: Int,
    val logGPS: Boolean,
    val logIMU: Boolean
)

data class StorageInfo(
    val path: String,
    val isPublic: Boolean,
    val accessInstructions: String,
    val type: String
)