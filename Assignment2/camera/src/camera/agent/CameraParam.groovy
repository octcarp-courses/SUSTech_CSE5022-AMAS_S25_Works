package camera.agent

import groovy.transform.CompileStatic

/**
 * For define camera creation parameters
 */
@CompileStatic
class CameraParam {
    final double x
    final double y
    final double rotation
    // final double radius
    // final double angle

    CameraParam(
    double x,
    double y,
    double rotation
    // double radius,
    // double angle
    ) {
        this.x = x
        this.y = y
        this.rotation = rotation
        // this.radius = radius
        // this.angle = angle
    }
}
