package camera.context

import camera.agent.CameraParam
import groovy.transform.CompileStatic

/**
 * Camera scenarios preset
 */
@CompileStatic
class CameraScenario {
    final static List<CameraScenario> scenarios = []

    final double worldX
    final double worldY
    final List<CameraParam> cameraParams = []

    private CameraScenario(double worldX, double worldY) {
        this.worldX = worldX
        this.worldY = worldY
    }

    static {
        init1()
    }

    private static void init1() {
        // Create scenario with world size (x size, y size)
        CameraScenario scenario = new CameraScenario(40, 30)
        def cp = scenario.cameraParams

        // Camera parameters: location x, location y, rotation angle
        cp << new CameraParam(5, 2, 90)
        cp << new CameraParam(15, 2, 90)
        cp << new CameraParam(25, 2, 90)
        cp << new CameraParam(35, 2, 90)
        cp << new CameraParam(5, 28, -90)
        cp << new CameraParam(15, 28, -90)
        cp << new CameraParam(25, 28, -90)
        cp << new CameraParam(35, 28, -90)

        // Add this scenario preset to preset list
        scenarios << scenario
    }
}
