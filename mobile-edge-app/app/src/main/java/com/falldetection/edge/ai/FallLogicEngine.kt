package com.falldetection.edge.ai

import com.falldetection.edge.model.BoundingBox
import com.falldetection.edge.model.FallState
import com.falldetection.edge.model.Keypoint
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

class FallLogicEngine(
    private val threshold: Float = 0.43f,
    private val onFallConfirmed: (trackId: Int, score: Float, angle: Float, aspect: Float, box: BoundingBox) -> Unit,
    private val onRecovered: (trackId: Int, angle: Float, aspect: Float) -> Unit
) {
    data class Sample(
        val keypoints: List<Keypoint>,
        val box: BoundingBox,
        val timestampMs: Long
    )

    private val personHistories = mutableMapOf<Int, PersonState>()

    class PersonState(val trackId: Int) {
        val window = ArrayDeque<Sample>(32)
        var currentState: FallState = FallState.NORMAL
        var suspectedStartTimeMs: Long = 0L
        var confirmedTimeMs: Long = 0L
        var lastFallScore: Float = 0f
    }

    fun updatePerson(
        trackId: Int,
        box: BoundingBox,
        keypoints: List<Keypoint>,
        currentTimeMs: Long
    ): FallState {
        val state = personHistories.getOrPut(trackId) { PersonState(trackId) }
        state.window.addLast(Sample(keypoints, box, currentTimeMs))
        if (state.window.size > 30) state.window.removeFirst()

        val bodyAngle = calculateBodyAngle(keypoints)
        val aspectRatio = box.aspectRatio

        // Heuristic checks
        val isHorizontal = aspectRatio >= 0.88f
        val isLyingAngle = bodyAngle > 50f
        val isStandingBent = aspectRatio < 0.75f && hasStandingLegs(keypoints)
        val isTrueLying = isLyingAngle && (isHorizontal || bodyAngle > 75f) && !isStandingBent

        when (state.currentState) {
            FallState.NORMAL -> {
                if (isTrueLying) {
                    state.currentState = FallState.FALL_SUSPECTED
                    state.suspectedStartTimeMs = currentTimeMs
                }
            }
            FallState.FALL_SUSPECTED -> {
                val elapsedSec = (currentTimeMs - state.suspectedStartTimeMs) / 1000f
                if (bodyAngle < 38f || aspectRatio < 0.70f) {
                    state.currentState = FallState.NORMAL
                } else if (elapsedSec >= 1.5f) {
                    if (isTrueLying) {
                        state.currentState = FallState.FALL_CONFIRMED
                        state.confirmedTimeMs = currentTimeMs
                        state.lastFallScore = 0.92f
                        onFallConfirmed(trackId, state.lastFallScore, bodyAngle, aspectRatio, box)
                    } else {
                        state.currentState = FallState.NORMAL
                    }
                }
            }
            FallState.FALL_CONFIRMED -> {
                // Hybrid Recovery Check
                if (bodyAngle < 35f && aspectRatio < 0.70f) {
                    state.currentState = FallState.NORMAL
                    onRecovered(trackId, bodyAngle, aspectRatio)
                }
            }
            else -> {}
        }

        return state.currentState
    }

    fun resetAll() {
        personHistories.clear()
    }

    private fun calculateBodyAngle(kps: List<Keypoint>): Float {
        if (kps.size < 13) return 0f
        val ls = kps[5]
        val rs = kps[6]
        val lh = kps[11]
        val rh = kps[12]

        if (ls.score < 0.2f || rs.score < 0.2f || lh.score < 0.2f || rh.score < 0.2f) return 0f

        val midShoulderX = (ls.x + rs.x) / 2f
        val midShoulderY = (ls.y + rs.y) / 2f
        val midHipX = (lh.x + rh.x) / 2f
        val midHipY = (lh.y + rh.y) / 2f

        val dx = abs(midHipX - midShoulderX)
        val dy = abs(midHipY - midShoulderY)
        return (atan2(dx.toDouble(), (dy + 1e-6).toDouble()) * 180.0 / Math.PI).toFloat()
    }

    private fun hasStandingLegs(kps: List<Keypoint>): Boolean {
        if (kps.size < 17) return false
        val lh = kps[11]
        val rh = kps[12]
        val la = kps[15]
        val ra = kps[16]
        if (la.score > 0.2f && ra.score > 0.2f) {
            val midHipY = (lh.y + rh.y) / 2f
            val midAnkleY = (la.y + ra.y) / 2f
            return (midAnkleY - midHipY) > 0.35f
        }
        return false
    }
}

