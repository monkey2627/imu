package dev.slimevr.unit

import com.jme3.math.FastMath
import dev.slimevr.VRServer.Companion.getNextLocalTrackerId
import dev.slimevr.tracking.trackers.Tracker
import dev.slimevr.tracking.trackers.udp.IMUType
import dev.slimevr.unit.TrackerTestUtils.vectorApproxEqual
import io.github.axisangles.ktmath.EulerAngles
import io.github.axisangles.ktmath.EulerOrder
import io.github.axisangles.ktmath.Quaternion
import io.github.axisangles.ktmath.Vector3
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class ReferenceAdjustedAccelerationTests {
	private fun tracker(): Tracker = Tracker(
		null,
		getNextLocalTrackerId(),
		"accel-test",
		"accel-test",
		null,
		hasRotation = true,
		hasAcceleration = true,
		imuType = IMUType.UNKNOWN,
		allowReset = true,
		allowMounting = true,
		trackRotDirection = false,
	)

	@Test
	fun `yaw reset rotates acceleration into the new reference heading`() {
		val tracker = tracker()
		val localAcceleration = Vector3(0f, 0f, 1f)
		val reference = EulerAngles(EulerOrder.YZX, 0f, FastMath.HALF_PI, 0f).toQuaternion()

		tracker.resetsHandler.mountingOrientation = Quaternion.IDENTITY
		tracker.setRotation(Quaternion.IDENTITY)
		tracker.setAcceleration(localAcceleration)
		tracker.resetsHandler.resetFull(Quaternion.IDENTITY)
		tracker.resetsHandler.resetYaw(reference)

		val expected = reference.sandwich(localAcceleration)
		val actual = tracker.getFullyReferenceAdjustedAcceleration()
		assertTrue(vectorApproxEqual(expected, actual, 1e-5f), "$expected != $actual")
	}

	@Test
	fun `manual mounting orientation does not rotate world acceleration`() {
		val tracker = tracker()
		val rawRotation = EulerAngles(EulerOrder.YZX, 0.2f, -0.4f, 0.3f).toQuaternion()
		val localAcceleration = Vector3(1.25f, -0.5f, 2.0f)

		tracker.setRotation(rawRotation)
		tracker.setAcceleration(localAcceleration)
		tracker.resetsHandler.mountingOrientation = Quaternion.IDENTITY
		val before = tracker.getFullyReferenceAdjustedAcceleration()

		tracker.resetsHandler.mountingOrientation =
			EulerAngles(EulerOrder.YZX, 0.7f, 1.1f, -0.2f).toQuaternion()
		val after = tracker.getFullyReferenceAdjustedAcceleration()

		assertTrue(vectorApproxEqual(before, after, 1e-5f), "$before != $after")
	}

	@Test
	fun `reference adjustment preserves acceleration magnitude`() {
		val tracker = tracker()
		val acceleration = Vector3(1.25f, -0.5f, 2.0f)
		val reference = EulerAngles(EulerOrder.YZX, 0f, 1.2f, 0f).toQuaternion()

		tracker.resetsHandler.mountingOrientation = Quaternion.IDENTITY
		tracker.setRotation(EulerAngles(EulerOrder.YZX, 0.2f, -0.4f, 0.3f).toQuaternion())
		tracker.setAcceleration(acceleration)
		tracker.resetsHandler.resetFull(reference)

		val expectedLength = tracker.getAcceleration().len()
		val actualLength = tracker.getFullyReferenceAdjustedAcceleration().len()
		assertTrue(FastMath.isApproxEqual(expectedLength, actualLength, 1e-5f))
	}
}
