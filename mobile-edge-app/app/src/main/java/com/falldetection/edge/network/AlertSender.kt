package com.falldetection.edge.network

import com.falldetection.edge.model.BoundingBox
import com.falldetection.edge.model.FallState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

class AlertSender(private val getBaseUrl: () -> String) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .writeTimeout(3, TimeUnit.SECONDS)
        .readTimeout(3, TimeUnit.SECONDS)
        .build()

    private val isoFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.getDefault())

    suspend fun sendFallAlert(
        cameraId: String,
        location: String,
        trackId: Int,
        fallScore: Float,
        bodyAngle: Float,
        aspectRatio: Float,
        box: BoundingBox,
        snapshotBase64: String? = null
    ): Boolean = withContext(Dispatchers.IO) {
        val root = JSONObject().apply {
            put("eventId", "evt_${System.currentTimeMillis()}")
            put("eventType", "FallConfirmed")
            put("severity", "Critical")
            put("timestamp", isoFormat.format(Date()))
            put("cameraId", cameraId)
            put("location", location)

            val person = JSONObject().apply {
                put("trackId", trackId)
                put("state", FallState.FALL_CONFIRMED.name)
                put("fallScore", fallScore.toDouble())
                put("bodyAngle", bodyAngle.toDouble())
                put("aspectRatio", aspectRatio.toDouble())
                put("boundingBox", JSONObject().apply {
                    put("x1", box.x1.toInt())
                    put("y1", box.y1.toInt())
                    put("x2", box.x2.toInt())
                    put("y2", box.y2.toInt())
                })
            }
            put("person", person)
            if (!snapshotBase64.isNullOrEmpty()) {
                put("snapshotBase64", "data:image/jpeg;base64,$snapshotBase64")
            }
        }

        postJson("${getBaseUrl().trimEnd('/')}/api/alerts/fall", root.toString())
    }

    suspend fun sendRecoveredAlert(
        cameraId: String,
        location: String,
        trackId: Int,
        bodyAngle: Float,
        aspectRatio: Float
    ): Boolean = withContext(Dispatchers.IO) {
        val root = JSONObject().apply {
            put("eventId", "evt_${System.currentTimeMillis()}")
            put("eventType", "Recovered")
            put("severity", "Info")
            put("timestamp", isoFormat.format(Date()))
            put("cameraId", cameraId)
            put("location", location)

            val person = JSONObject().apply {
                put("trackId", trackId)
                put("state", FallState.NORMAL.name)
                put("fallScore", 0.05)
                put("bodyAngle", bodyAngle.toDouble())
                put("aspectRatio", aspectRatio.toDouble())
            }
            put("person", person)
        }

        postJson("${getBaseUrl().trimEnd('/')}/api/alerts/recovered", root.toString())
    }

    suspend fun testConnection(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("${getBaseUrl().trimEnd('/')}/api/alerts/test")
                .get()
                .build()
            client.newCall(request).execute().use { it.isSuccessful }
        } catch (_: Exception) {
            false
        }
    }

    private fun postJson(url: String, json: String): Boolean {
        return try {
            val body = json.toRequestBody("application/json; charset=utf-8".toMediaType())
            val request = Request.Builder().url(url).post(body).build()
            client.newCall(request).execute().use { it.isSuccessful }
        } catch (_: Exception) {
            false
        }
    }
}

