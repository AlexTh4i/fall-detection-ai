package com.falldetection.edge.model

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
) {
    val width: Float get() = Math.max(0f, x2 - x1)
    val height: Float get() = Math.max(0f, y2 - y1)
    val area: Float get() = width * height
    val aspectRatio: Float get() = width / Math.max(height, 1e-6f)
    val centerX: Float get() = (x1 + x2) / 2f
    val centerY: Float get() = (y1 + y2) / 2f
}

data class Keypoint(
    val x: Float,
    val y: Float,
    val score: Float
)

data class DetectionResult(
    val trackId: Int,
    val box: BoundingBox,
    val keypoints: List<Keypoint>, // 17 COCO keypoints
    val confidence: Float
)

enum class FallState {
    NORMAL,
    FALL_SUSPECTED,
    FALL_CONFIRMED,
    UNKNOWN
}

