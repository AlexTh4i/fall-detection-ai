package com.falldetection.edge

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.falldetection.edge.ai.FallLogicEngine
import com.falldetection.edge.ai.LightweightTracker
import com.falldetection.edge.data.AppSettings
import com.falldetection.edge.databinding.ActivityMainBinding
import com.falldetection.edge.databinding.DialogSettingsBinding
import com.falldetection.edge.model.BoundingBox
import com.falldetection.edge.model.FallState
import com.falldetection.edge.model.Keypoint
import com.falldetection.edge.network.AlertSender
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: AppSettings
    private lateinit var alertSender: AlertSender
    private lateinit var tracker: LightweightTracker
    private lateinit var fallEngine: FallLogicEngine

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val uiHandler = Handler(Looper.getMainLooper())
    private var toneGenerator: ToneGenerator? = null

    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val dateFormat = SimpleDateFormat("dd/MM/yyyy", Locale.getDefault())

    private var currentOverallState: FallState = FallState.NORMAL

    private val clockRunnable = object : Runnable {
        override fun run() {
            val now = Date()
            val dayOfWeek = when (java.util.Calendar.getInstance().get(java.util.Calendar.DAY_OF_WEEK)) {
                java.util.Calendar.SUNDAY -> "CN"
                java.util.Calendar.MONDAY -> "T2"
                java.util.Calendar.TUESDAY -> "T3"
                java.util.Calendar.WEDNESDAY -> "T4"
                java.util.Calendar.THURSDAY -> "T5"
                java.util.Calendar.FRIDAY -> "T6"
                else -> "T7"
            }
            binding.tvClock.text = "$dayOfWeek ${dateFormat.format(now)} ${timeFormat.format(now)}"
            uiHandler.postDelayed(this, 1000)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Cần cấp quyền Camera để nhận diện!", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        settings = AppSettings(this)
        alertSender = AlertSender { settings.serverUrl }
        tracker = LightweightTracker()

        toneGenerator = try {
            ToneGenerator(AudioManager.STREAM_ALARM, 100)
        } catch (_: Exception) {
            null
        }

        fallEngine = FallLogicEngine(
            threshold = 0.43f,
            onFallConfirmed = { trackId, score, angle, aspect, box ->
                onFallDetected(trackId, score, angle, aspect, box)
            },
            onRecovered = { trackId, angle, aspect ->
                onPersonRecovered(trackId, angle, aspect)
            }
        )

        updateUiFromSettings()
        setupListeners()
        uiHandler.post(clockRunnable)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        checkServerConnection()
    }

    private fun updateUiFromSettings() {
        binding.tvCamera.text = "${settings.cameraName} · ${settings.roomName}"
        binding.tvLocation.text = settings.location
    }

    private fun setupListeners() {
        binding.btnSettings.setOnClickListener {
            showSettingsDialog()
        }
    }

    private fun showSettingsDialog() {
        val dialogBinding = DialogSettingsBinding.inflate(layoutInflater)
        dialogBinding.etCamera.setText(settings.cameraName)
        dialogBinding.etCameraId.setText(settings.cameraId)
        dialogBinding.etRoom.setText(settings.roomName)
        dialogBinding.etLocation.setText(settings.location)
        dialogBinding.etServer.setText(settings.serverUrl)
        dialogBinding.cbFront.isChecked = settings.useFrontCamera
        dialogBinding.cbSiren.isChecked = settings.enableSiren

        val dialog = AlertDialog.Builder(this)
            .setTitle("Cài đặt Camera & Hệ thống")
            .setView(dialogBinding.root)
            .setPositiveButton("Lưu") { _, _ ->
                val prevFront = settings.useFrontCamera
                settings.cameraName = dialogBinding.etCamera.text.toString().trim()
                settings.cameraId = dialogBinding.etCameraId.text.toString().trim()
                settings.roomName = dialogBinding.etRoom.text.toString().trim()
                settings.location = dialogBinding.etLocation.text.toString().trim()
                settings.serverUrl = dialogBinding.etServer.text.toString().trim()
                settings.useFrontCamera = dialogBinding.cbFront.isChecked
                settings.enableSiren = dialogBinding.cbSiren.isChecked

                updateUiFromSettings()
                checkServerConnection()

                if (prevFront != settings.useFrontCamera) {
                    startCamera()
                }
            }
            .setNegativeButton("Hủy", null)
            .create()

        dialogBinding.btnTest.setOnClickListener {
            lifecycleScope.launch {
                val ok = alertSender.testConnection()
                Toast.makeText(
                    this@MainActivity,
                    if (ok) "Kết nối C# Server thành công!" else "Không kết nối được Server!",
                    Toast.LENGTH_SHORT
                ).show()
                updateServerStatus(ok)
            }
        }

        dialogBinding.btnReset.setOnClickListener {
            resetFallState()
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                // Image frame processed here by ONNX Runtime inference
                imageProxy.close()
            }

            val cameraSelector = if (settings.useFrontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFallDetected(
        trackId: Int,
        score: Float,
        angle: Float,
        aspect: Float,
        box: BoundingBox
    ) {
        runOnUiThread {
            currentOverallState = FallState.FALL_CONFIRMED
            binding.tvStatus.text = "🚨 ĐANG TÉ NGÃ — ĐÃ GỬI CẢNH BÁO"
            binding.tvStatus.setBackgroundColor(0xCCD32F2F.toInt())
            binding.alertTint.visibility = View.VISIBLE

            if (settings.enableSiren) {
                toneGenerator?.startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 1500)
            }
        }

        lifecycleScope.launch {
            alertSender.sendFallAlert(
                cameraId = settings.cameraId,
                location = settings.location,
                trackId = trackId,
                fallScore = score,
                bodyAngle = angle,
                aspectRatio = aspect,
                box = box,
                snapshotBase64 = null
            )
        }
    }

    private fun onPersonRecovered(trackId: Int, angle: Float, aspect: Float) {
        runOnUiThread {
            currentOverallState = FallState.NORMAL
            binding.tvStatus.text = "✔ Bình thường"
            binding.tvStatus.setBackgroundColor(0x992E7D32.toInt())
            binding.alertTint.visibility = View.GONE
        }

        lifecycleScope.launch {
            alertSender.sendRecoveredAlert(
                cameraId = settings.cameraId,
                location = settings.location,
                trackId = trackId,
                bodyAngle = angle,
                aspectRatio = aspect
            )
        }
    }

    private fun resetFallState() {
        fallEngine.resetAll()
        currentOverallState = FallState.NORMAL
        binding.tvStatus.text = "✔ Bình thường"
        binding.tvStatus.setBackgroundColor(0x992E7D32.toInt())
        binding.alertTint.visibility = View.GONE
    }

    private fun checkServerConnection() {
        lifecycleScope.launch {
            val ok = alertSender.testConnection()
            updateServerStatus(ok)
        }
    }

    private fun updateServerStatus(connected: Boolean) {
        binding.tvServer.text = if (connected) "● Server (Online)" else "○ Server (Offline)"
        binding.tvServer.setTextColor(if (connected) 0xFF4ADE80.toInt() else 0xFFEF4444.toInt())
    }

    override fun onDestroy() {
        super.onDestroy()
        uiHandler.removeCallbacks(clockRunnable)
        cameraExecutor.shutdown()
        toneGenerator?.release()
    }
}

