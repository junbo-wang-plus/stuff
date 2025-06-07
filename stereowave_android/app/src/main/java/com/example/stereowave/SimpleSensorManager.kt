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

class SimpleSensorManager(private val context: Context) : SensorEventListener, LocationListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

    private var gpsWriter: PrintWriter? = null
    private var imuWriter: PrintWriter? = null
    private var isLogging = false

    // Sensors
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    fun startGPSLogging(sessionDir: File) {
        try {
            val gpsFile = File(sessionDir, "gps_data.csv")
            gpsWriter = PrintWriter(FileWriter(gpsFile))

            // Write CSV header
            gpsWriter?.println("timestamp,latitude,longitude,altitude,accuracy,bearing,speed,satellites")

            // Request location updates
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000, // 1 second intervals
                    0f,   // No minimum distance
                    this
                )
                Log.d("SimpleSensorManager", "GPS logging started")
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
            val imuFile = File(sessionDir, "imu_data.csv")
            imuWriter = PrintWriter(FileWriter(imuFile))

            // Write CSV header
            imuWriter?.println("timestamp,sensor_type,x,y,z,accuracy")

            // Register sensor listeners at 50Hz (20ms intervals)
            val samplingPeriod = SensorManager.SENSOR_DELAY_UI // ~20ms

            accelerometer?.let {
                sensorManager.registerListener(this, it, samplingPeriod)
                Log.d("SimpleSensorManager", "Accelerometer logging started")
            }

            gyroscope?.let {
                sensorManager.registerListener(this, it, samplingPeriod)
                Log.d("SimpleSensorManager", "Gyroscope logging started")
            }

            magnetometer?.let {
                sensorManager.registerListener(this, it, samplingPeriod)
                Log.d("SimpleSensorManager", "Magnetometer logging started")
            }

            isLogging = true

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Failed to start IMU logging", e)
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

        // Stop IMU
        try {
            sensorManager.unregisterListener(this)
            imuWriter?.close()
            imuWriter = null
            Log.d("SimpleSensorManager", "IMU logging stopped")
        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error stopping IMU", e)
        }
    }

    // LocationListener implementation
    override fun onLocationChanged(location: Location) {
        if (!isLogging) return

        try {
            val timestamp = System.currentTimeMillis()
            gpsWriter?.println(
                "$timestamp," +
                        "${location.latitude}," +
                        "${location.longitude}," +
                        "${location.altitude}," +
                        "${location.accuracy}," +
                        "${location.bearing}," +
                        "${location.speed}," +
                        "0" // Satellites count not available in standard API
            )

            Log.d("SimpleSensorManager", "GPS: ${location.latitude}, ${location.longitude}, acc=${location.accuracy}m")

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

    // SensorEventListener implementation
    override fun onSensorChanged(event: SensorEvent) {
        if (!isLogging) return

        try {
            val timestamp = System.currentTimeMillis()
            val sensorType = when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> "accelerometer"
                Sensor.TYPE_GYROSCOPE -> "gyroscope"
                Sensor.TYPE_MAGNETIC_FIELD -> "magnetometer"
                else -> "unknown"
            }

            imuWriter?.println(
                "$timestamp," +
                        "$sensorType," +
                        "${event.values[0]}," +
                        "${event.values[1]}," +
                        "${event.values[2]}," +
                        "${event.accuracy}"
            )

        } catch (e: Exception) {
            Log.e("SimpleSensorManager", "Error writing IMU data", e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        val sensorName = when (sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> "Accelerometer"
            Sensor.TYPE_GYROSCOPE -> "Gyroscope"
            Sensor.TYPE_MAGNETIC_FIELD -> "Magnetometer"
            else -> "Unknown sensor"
        }
        Log.d("SimpleSensorManager", "$sensorName accuracy changed: $accuracy")
    }
}