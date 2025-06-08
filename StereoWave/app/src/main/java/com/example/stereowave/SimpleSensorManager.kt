package com.example.stereowave

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.util.Log
import kotlinx.coroutines.*
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import kotlin.math.*

class SimpleSensorManager(private val context: Context) : SensorEventListener, LocationListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    // File writers for different data streams
    private var gpsWriter: PrintWriter? = null
    private var orientationWriter: PrintWriter? = null
    private var motionWriter: PrintWriter? = null
    private var rawImuWriter: PrintWriter? = null
    private var isLogging = false

    // Raw sensors (for backup/validation)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    // Android sensor fusion sensors (primary for WASS)
    private val rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    private val gameRotation = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

    // Current sensor state for motion prediction
    private var lastOrientation = FloatArray(3) // roll, pitch, yaw
    private var lastLinearAccel = FloatArray(3)
    private var lastAngularVel = FloatArray(3)

    fun startGPSLogging(sessionDir: File) {
        try {
            val gpsFile = File(sessionDir, "gps_data.csv")
            gpsWriter = PrintWriter(FileWriter(gpsFile))

            // Write CSV header with additional WASS-useful fields
            gpsWriter?.println("timestamp_ms,latitude_deg,longitude_deg,altitude_m,accuracy_m,bearing_deg,speed_mps,hdop,vdop,satellites,fix_quality")

            // Request high-accuracy location updates
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000, // 1 second intervals
                    0f,   // No minimum distance
                    this
                )
                Log.d("SimpleSensorManager", "Enhanced GPS logging started")
            } else {
                Log.w("SimpleSensorManager", "GPS provider not enabled")
            }

        } catch (e: SecurityException) {
            Log.e("SimpleSensorManager", "GPS permission not granted", e)
        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Failed to start GPS logging", e)
        }
    }

    fun startIMULogging(sessionDir: File) {
        try {
            // Create separate files for different data types (easier WASS processing)
            createOrientationLog(sessionDir)
            createMotionLog(sessionDir)
            createRawIMULog(sessionDir)
            createIMUMetadata(sessionDir)

            // Register Android sensor fusion listeners (higher priority)
            val fastSampling = SensorManager.SENSOR_DELAY_FASTEST // ~5ms for critical sensors
            val normalSampling = SensorManager.SENSOR_DELAY_UI   // ~20ms for backup sensors

            // Primary orientation sensor (quaternion-based)
            rotationVector?.let {
                sensorManager.registerListener(this, it, fastSampling)
                Log.d("SimpleSensorManager", "Rotation vector (orientation) logging started at ~5ms")
            }

            // Motion sensors (gravity-compensated)
            gravity?.let {
                sensorManager.registerListener(this, it, fastSampling)
                Log.d("SimpleSensorManager", "Gravity vector logging started")
            }

            linearAcceleration?.let {
                sensorManager.registerListener(this, it, fastSampling)
                Log.d("SimpleSensorManager", "Linear acceleration (motion) logging started")
            }

            // Game rotation vector (no magnetometer - more stable for short-term)
            gameRotation?.let {
                sensorManager.registerListener(this, it, fastSampling)
                Log.d("SimpleSensorManager", "Game rotation vector logging started")
            }

            // Raw sensors for validation/backup
            accelerometer?.let {
                sensorManager.registerListener(this, it, normalSampling)
                Log.d("SimpleSensorManager", "Raw accelerometer logging started")
            }

            gyroscope?.let {
                sensorManager.registerListener(this, it, normalSampling)
                Log.d("SimpleSensorManager", "Raw gyroscope logging started")
            }

            magnetometer?.let {
                sensorManager.registerListener(this, it, normalSampling)
                Log.d("SimpleSensorManager", "Raw magnetometer logging started")
            }

            isLogging = true

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Failed to start enhanced IMU logging", e)
        }
    }

    private fun createOrientationLog(sessionDir: File) {
        val orientationFile = File(sessionDir, "orientation_data.csv")
        orientationWriter = PrintWriter(FileWriter(orientationFile))

        // WASS-friendly orientation data
        orientationWriter?.println("timestamp_ms,roll_deg,pitch_deg,yaw_deg,quat_x,quat_y,quat_z,quat_w,accuracy,sensor_type")
    }

    private fun createMotionLog(sessionDir: File) {
        val motionFile = File(sessionDir, "motion_data.csv")
        motionWriter = PrintWriter(FileWriter(motionFile))

        // WASS-friendly motion data (gravity compensated)
        motionWriter?.println("timestamp_ms,linear_accel_x_mps2,linear_accel_y_mps2,linear_accel_z_mps2,gravity_x_mps2,gravity_y_mps2,gravity_z_mps2,angular_vel_x_radps,angular_vel_y_radps,angular_vel_z_radps,accuracy")
    }

    private fun createRawIMULog(sessionDir: File) {
        val rawFile = File(sessionDir, "raw_imu_data.csv")
        rawImuWriter = PrintWriter(FileWriter(rawFile))

        // Raw sensor backup data
        rawImuWriter?.println("timestamp_ms,sensor_type,x,y,z,accuracy")
    }

    private fun createIMUMetadata(sessionDir: File) {
        try {
            val metadataFile = File(sessionDir, "imu_metadata.json")
            val metadata = StringBuilder()
            metadata.append("{\n")
            metadata.append("  \"coordinate_system\": \"Android_device_coordinates\",\n")
            metadata.append("  \"orientation_convention\": \"roll_pitch_yaw_degrees\",\n")
            metadata.append("  \"coordinate_axes\": {\n")
            metadata.append("    \"x\": \"device_right_positive\",\n")
            metadata.append("    \"y\": \"device_forward_positive\",\n")
            metadata.append("    \"z\": \"device_up_positive\"\n")
            metadata.append("  },\n")
            metadata.append("  \"rotation_convention\": \"right_hand_rule\",\n")
            metadata.append("  \"units\": {\n")
            metadata.append("    \"acceleration\": \"meters_per_second_squared\",\n")
            metadata.append("    \"angular_velocity\": \"radians_per_second\",\n")
            metadata.append("    \"orientation\": \"degrees\",\n")
            metadata.append("    \"quaternion\": \"normalized_wxyz\"\n")
            metadata.append("  },\n")
            metadata.append("  \"sampling_rates\": {\n")
            metadata.append("    \"orientation_motion\": \"~200Hz (5ms)\",\n")
            metadata.append("    \"raw_sensors\": \"~50Hz (20ms)\"\n")
            metadata.append("  },\n")
            metadata.append("  \"notes\": \"For WASS processing, use orientation_data.csv and motion_data.csv. Raw data is backup/validation.\"\n")
            metadata.append("}")

            FileWriter(metadataFile).use { writer ->
                writer.write(metadata.toString())
            }

            Log.d("SimpleSensorManager", "IMU metadata created for WASS processing")

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Failed to create IMU metadata", e)
        }
    }

    fun stopLogging() {
        isLogging = false

        // Stop GPS
        try {
            locationManager.removeUpdates(this)
            gpsWriter?.close()
            gpsWriter = null
            Log.d("SimpleSensorManager", "GPS logging stopped")
        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error stopping GPS", e)
        }

        // Stop all IMU sensors
        try {
            sensorManager.unregisterListener(this)
            orientationWriter?.close()
            motionWriter?.close()
            rawImuWriter?.close()
            orientationWriter = null
            motionWriter = null
            rawImuWriter = null
            Log.d("SimpleSensorManager", "Enhanced IMU logging stopped")
        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error stopping IMU", e)
        }
    }

    // LocationListener implementation (enhanced)
    override fun onLocationChanged(location: Location) {
        if (!isLogging) return

        try {
            val timestamp = System.currentTimeMillis()

            // Extract additional GPS quality metrics when available (capture extras to avoid smart cast issues)
            val extras = location.extras
            val hdop = if (extras?.containsKey("hdop") == true) {
                extras.getFloat("hdop")
            } else -1f

            val vdop = if (extras?.containsKey("vdop") == true) {
                extras.getFloat("vdop")
            } else -1f

            val satellites = if (extras?.containsKey("satellites") == true) {
                extras.getInt("satellites")
            } else -1

            gpsWriter?.println(
                "$timestamp," +
                        "${location.latitude}," +
                        "${location.longitude}," +
                        "${location.altitude}," +
                        "${location.accuracy}," +
                        "${location.bearing}," +
                        "${location.speed}," +
                        "$hdop," +
                        "$vdop," +
                        "$satellites," +
                        "3D" // Assume 3D fix for modern devices
            )

            Log.d("SimpleSensorManager", "Enhanced GPS: ${location.latitude}, ${location.longitude}, acc=${location.accuracy}m")

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error writing GPS data", e)
        }
    }

    override fun onProviderEnabled(provider: String) {
        Log.d("SimpleSensorManager", "Location provider enabled: $provider")
    }

    override fun onProviderDisabled(provider: String) {
        Log.w("SimpleSensorManager", "Location provider disabled: $provider")
    }

    @Deprecated("Deprecated in API level 29")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {
        // No action needed
    }

    // Enhanced SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        if (!isLogging) return

        try {
            val timestamp = System.currentTimeMillis()

            when (event.sensor.type) {
                Sensor.TYPE_ROTATION_VECTOR -> {
                    handleRotationVector(timestamp, event)
                }
                Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                    handleGameRotationVector(timestamp, event)
                }
                Sensor.TYPE_GRAVITY -> {
                    handleGravityVector(timestamp, event)
                }
                Sensor.TYPE_LINEAR_ACCELERATION -> {
                    handleLinearAcceleration(timestamp, event)
                }
                // Raw sensors (backup data)
                Sensor.TYPE_ACCELEROMETER -> {
                    rawImuWriter?.println("$timestamp,accelerometer,${event.values[0]},${event.values[1]},${event.values[2]},${event.accuracy}")
                }
                Sensor.TYPE_GYROSCOPE -> {
                    rawImuWriter?.println("$timestamp,gyroscope,${event.values[0]},${event.values[1]},${event.values[2]},${event.accuracy}")
                    // Update angular velocity for motion prediction
                    lastAngularVel[0] = event.values[0]
                    lastAngularVel[1] = event.values[1]
                    lastAngularVel[2] = event.values[2]
                }
                Sensor.TYPE_MAGNETIC_FIELD -> {
                    rawImuWriter?.println("$timestamp,magnetometer,${event.values[0]},${event.values[1]},${event.values[2]},${event.accuracy}")
                }
            }

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error processing sensor data", e)
        }
    }

    private fun handleRotationVector(timestamp: Long, event: SensorEvent) {
        // Convert quaternion to Euler angles for WASS compatibility
        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)

        val roll = atan2(2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
            1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])) * 180 / PI
        val pitch = asin(2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1])) * 180 / PI
        val yaw = atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1 - 2 * (quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3])) * 180 / PI

        // Update current orientation state
        lastOrientation[0] = roll.toFloat()
        lastOrientation[1] = pitch.toFloat()
        lastOrientation[2] = yaw.toFloat()

        // Write to orientation file (WASS primary input)
        orientationWriter?.println(
            "$timestamp," +
                    "$roll," +
                    "$pitch," +
                    "$yaw," +
                    "${quaternion[1]}," + // x
                    "${quaternion[2]}," + // y
                    "${quaternion[3]}," + // z
                    "${quaternion[0]}," + // w
                    "${event.accuracy}," +
                    "rotation_vector"
        )
    }

    private fun handleGameRotationVector(timestamp: Long, event: SensorEvent) {
        // Game rotation vector (no magnetometer - more stable short-term)
        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)

        val roll = atan2(2 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
            1 - 2 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])) * 180 / PI
        val pitch = asin(2 * (quaternion[0] * quaternion[2] - quaternion[3] * quaternion[1])) * 180 / PI
        val yaw = atan2(2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
            1 - 2 * (quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3])) * 180 / PI

        orientationWriter?.println(
            "$timestamp," +
                    "$roll," +
                    "$pitch," +
                    "$yaw," +
                    "${quaternion[1]}," +
                    "${quaternion[2]}," +
                    "${quaternion[3]}," +
                    "${quaternion[0]}," +
                    "${event.accuracy}," +
                    "game_rotation"
        )
    }

    private fun handleGravityVector(timestamp: Long, event: SensorEvent) {
        // Write gravity vector to motion file (for WASS coordinate transformation)
        motionWriter?.println(
            "$timestamp," +
                    "${lastLinearAccel[0]}," +
                    "${lastLinearAccel[1]}," +
                    "${lastLinearAccel[2]}," +
                    "${event.values[0]}," +
                    "${event.values[1]}," +
                    "${event.values[2]}," +
                    "${lastAngularVel[0]}," +
                    "${lastAngularVel[1]}," +
                    "${lastAngularVel[2]}," +
                    "${event.accuracy}"
        )
    }

    private fun handleLinearAcceleration(timestamp: Long, event: SensorEvent) {
        // Update linear acceleration state (gravity already removed by Android)
        lastLinearAccel[0] = event.values[0]
        lastLinearAccel[1] = event.values[1]
        lastLinearAccel[2] = event.values[2]

        // Motion data gets written in handleGravityVector for synchronization
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        val sensorName = when (sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> "Rotation Vector"
            Sensor.TYPE_GAME_ROTATION_VECTOR -> "Game Rotation"
            Sensor.TYPE_GRAVITY -> "Gravity"
            Sensor.TYPE_LINEAR_ACCELERATION -> "Linear Acceleration"
            Sensor.TYPE_ACCELEROMETER -> "Accelerometer"
            Sensor.TYPE_GYROSCOPE -> "Gyroscope"
            Sensor.TYPE_MAGNETIC_FIELD -> "Magnetometer"
            else -> "Unknown sensor"
        }
        Log.d("SimpleSensorManager", "$sensorName accuracy changed: $accuracy")
    }

    /**
     * Get current motion state for real-time camera compensation
     */
    fun getCurrentMotionState(): MotionState {
        return MotionState(
            orientation = lastOrientation.clone(),
            linearAcceleration = lastLinearAccel.clone(),
            angularVelocity = lastAngularVel.clone(),
            timestamp = System.currentTimeMillis()
        )
    }
}

/**
 * Current device motion state for real-time processing
 */
data class MotionState(
    val orientation: FloatArray,      // roll, pitch, yaw in degrees
    val linearAcceleration: FloatArray, // m/s² (gravity compensated)
    val angularVelocity: FloatArray,  // rad/s
    val timestamp: Long               // milliseconds
)