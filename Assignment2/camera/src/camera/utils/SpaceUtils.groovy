package camera.utils

import groovy.transform.CompileStatic

import repast.simphony.space.continuous.NdPoint

/**
 * Utils for 2D space item
 */
@CompileStatic
class SpaceUtils {
    static double[] calcDxDyDistance(NdPoint a, NdPoint b) {
        double dx = a.x - b.x
        double dy = a.y - b.y
        double distance = Math.hypot(dx, dy)
        return [dx, dy, distance] as double[]
    }
}
