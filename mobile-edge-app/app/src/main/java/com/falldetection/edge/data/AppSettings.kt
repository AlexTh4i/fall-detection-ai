package com.falldetection.edge.data

import android.content.Context
import android.content.SharedPreferences

class AppSettings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("fall_guard_prefs", Context.MODE_PRIVATE)

    var cameraName: String
        get() = prefs.getString("camera_name", "CAMERA 01") ?: "CAMERA 01"
        set(value) = prefs.edit().putString("camera_name", value).apply()

    var cameraId: String
        get() = prefs.getString("camera_id", "CAM_01") ?: "CAM_01"
        set(value) = prefs.edit().putString("camera_id", value).apply()

    var roomName: String
        get() = prefs.getString("room_name", "Phòng Khách") ?: "Phòng Khách"
        set(value) = prefs.edit().putString("room_name", value).apply()

    var location: String
        get() = prefs.getString("location", "Đà Nẵng, Việt Nam") ?: "Đà Nẵng, Việt Nam"
        set(value) = prefs.edit().putString("location", value).apply()

    var serverUrl: String
        get() = prefs.getString("server_url", "http://192.168.1.15:5000") ?: "http://192.168.1.15:5000"
        set(value) = prefs.edit().putString("server_url", value).apply()

    var useFrontCamera: Boolean
        get() = prefs.getBoolean("use_front_cam", false)
        set(value) = prefs.edit().putBoolean("use_front_cam", value).apply()

    var enableSiren: Boolean
        get() = prefs.getBoolean("enable_siren", true)
        set(value) = prefs.edit().putBoolean("enable_siren", value).apply()
}

