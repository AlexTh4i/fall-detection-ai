package com.falldetection.edge.ai

import com.falldetection.edge.model.BoundingBox
import com.falldetection.edge.model.DetectionResult
import com.falldetection.edge.model.Keypoint
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min

class LightweightTracker {
    private val nextId = AtomicInteger(1)
    private val activeTracks = mutableMapOf<Int, TrackedPerson>()

    data class TrackedPerson(
        val trackId: Int,
        var lastBox: BoundingBox,
        var lastSeenTimeMs: Long
    )

    fun update(detections: List<RawDetection>, currentTimeMs: Long): List<DetectionResult> {
        // Clean old tracks (inactive > 4 seconds)
        activeTracks.entries.removeIf { currentTimeMs - it.value.lastSeenTimeMs > 4000 }

        val assigned = mutableSetOf<Int>()
        val results = mutableListOf<DetectionResult>()

        for (det in detections) {
            var bestId: Int? = null
            var bestIou = 0.15f

            for ((id, trk) in activeTracks) {
                if (id in assigned) continue
                val iouVal = calculateIou(det.box, trk.lastBox)
                if (iouVal > bestIou) {
                    bestIou = iouVal
                    bestId = id
                }
            }

            val finalId = if (bestId != null) {
                assigned.add(bestId)
                activeTracks[bestId]!!.lastBox = det.box
                activeTracks[bestId]!!.lastSeenTimeMs = currentTimeMs
                bestId
            } else {
                val newId = nextId.getAndIncrement()
                activeTracks[newId] = TrackedPerson(newId, det.box, currentTimeMs)
                newId
            }

            results.add(
                DetectionResult(
                    trackId = finalId,
                    box = det.box,
                    keypoints = det.keypoints,
                    confidence = det.confidence
                )
            )
        }

        return results
    }

    private fun calculateIou(a: BoundingBox, b: BoundingBox): Float {
        val xA = max(a.x1, b.x1)
        val yA = max(a.y1, b.y1)
        val xB = min(a.x2, b.x2)
        val yB = min(a.y2, b.y2)

        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val boxAArea = a.area
        val boxBArea = b.area
        return interArea / max(boxAArea + boxBArea - interArea, 1e-6f)
    }

    data class RawDetection(
        val box: BoundingBox,
        val keypoints: List<Keypoint>,
        val confidence: Float
    )
}

