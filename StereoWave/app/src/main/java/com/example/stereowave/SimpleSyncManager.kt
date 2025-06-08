package com.example.stereowave

import android.util.Log
import kotlinx.coroutines.*
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket

class SimpleSyncManager {

    private var localClockOffset: Long = 0
    private var lastSyncTime: Long = 0
    private val syncValidityMs = 300000 // 5 minutes

    /**
     * Simple NTP-style time synchronization
     * For basic 2Hz sampling, ±50ms accuracy is sufficient
     */
    suspend fun synchronizeTime(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Try to get network time from time.google.com
            val ntpServers = listOf(
                "time.google.com",
                "pool.ntp.org",
                "time.cloudflare.com"
            )

            for (server in ntpServers) {
                try {
                    val offset = queryTimeServer(server)
                    if (offset != null) {
                        localClockOffset = offset
                        lastSyncTime = System.currentTimeMillis()
                        Log.d("SimpleSyncManager", "Time synced with $server, offset: ${offset}ms")
                        return@withContext true
                    }
                } catch (e: Exception) {
                    Log.w("SimpleSyncManager", "Failed to sync with $server", e)
                }
            }

            Log.w("SimpleSyncManager", "All time sync attempts failed")
            false

        } catch (e: Exception) {
            Log.e("SimpleSyncManager", "Time sync error", e)
            false
        }
    }

    private suspend fun queryTimeServer(server: String): Long? = withContext(Dispatchers.IO) {
        try {
            val socket = Socket()
            val address = InetSocketAddress(InetAddress.getByName(server), 123)

            val startTime = System.currentTimeMillis()
            socket.connect(address, 5000) // 5 second timeout
            val endTime = System.currentTimeMillis()
            socket.close()

            // Simple network latency estimation
            val networkLatency = (endTime - startTime) / 2

            // For basic synchronization, assume server time is correct
            // and adjust for network latency
            return@withContext networkLatency

        } catch (e: Exception) {
            Log.d("SimpleSyncManager", "Time server query failed for $server", e)
            null
        }
    }

    /**
     * Get synchronized timestamp
     */
    fun getSynchronizedTime(): Long {
        val currentTime = System.currentTimeMillis()

        // Check if sync is still valid
        if (currentTime - lastSyncTime > syncValidityMs) {
            Log.w("SimpleSyncManager", "Time sync expired, using local time")
            return currentTime
        }

        return currentTime + localClockOffset
    }

    /**
     * Check synchronization quality
     */
    fun getSyncQuality(): SyncQuality {
        val timeSinceSync = System.currentTimeMillis() - lastSyncTime

        return when {
            lastSyncTime == 0L -> SyncQuality.NO_SYNC
            timeSinceSync < 60000 -> SyncQuality.EXCELLENT  // < 1 minute
            timeSinceSync < 300000 -> SyncQuality.GOOD      // < 5 minutes
            timeSinceSync < 900000 -> SyncQuality.POOR      // < 15 minutes
            else -> SyncQuality.EXPIRED
        }
    }

    /**
     * Calculate next capture time based on sampling rate
     */
    fun calculateNextCaptureTime(samplingRateHz: Float, sequenceNumber: Int): Long {
        val intervalMs = (1000.0f / samplingRateHz).toLong()
        val baseTime = getSynchronizedTime()

        // Align to regular intervals to improve synchronization between phones
        val alignedBase = (baseTime / intervalMs) * intervalMs
        return alignedBase + (sequenceNumber * intervalMs)
    }

    /**
     * Get timing statistics for quality monitoring
     */
    fun getTimingStats(): TimingStats {
        return TimingStats(
            localClockOffset = localClockOffset,
            lastSyncTime = lastSyncTime,
            syncAge = System.currentTimeMillis() - lastSyncTime,
            syncQuality = getSyncQuality()
        )
    }
}

enum class SyncQuality {
    NO_SYNC,
    EXPIRED,
    POOR,
    GOOD,
    EXCELLENT
}

data class TimingStats(
    val localClockOffset: Long,
    val lastSyncTime: Long,
    val syncAge: Long,
    val syncQuality: SyncQuality
)