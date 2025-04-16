package camera.utils

import groovy.transform.CompileStatic

import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.parameter.Parameters

@CompileStatic
class ParameterUtils {
    private static ParameterUtils instance = null;

    private ParameterUtils() {}

    static ParameterUtils getInstance() {
        if (!instance) {
            synchronized (ParameterUtils) {
                if (!instance) instance = new ParameterUtils()
            }
        }
        instance
    }

    static void refresh() {
        instance = new ParameterUtils()
    }

    private final Parameters parameters = RunEnvironment.instance.parameters

    final double SPACE_X_SIZE = getDouble("SPACE_X_SIZE", 50)
    final double SPACE_Y_SIZE = getDouble("SPACE_Y_SIZE", 50)

    final int NUM_CAMERAS = getInt("NUM_CAMERAS", 5)
    final int NUM_TARGET_OBJS = getInt("NUM_TARGET_OBJS", 10)

    final double CAMERA_RADIUS = getDouble("CAMERA_RADIUS", 10)
    final double CAMERA_ANGLE = getDouble("CAMERA_ANGLE", 90)

    int getInt(String paramName, int defaultValue = 0) {
        try {
            parameters.getInteger(paramName)
        } catch (Exception e) {
            System.err.println "Parameter $paramName not found, using default value: $defaultValue"
            defaultValue
        }
    }

    double getDouble(String paramName, double defaultValue = 0.0) {
        try {
            parameters.getDouble(paramName)
        } catch (Exception e) {
            System.err.println "Parameter $paramName not found, using default value: $defaultValue"
            defaultValue
        }
    }

    String getString(String paramName, String defaultValue = "") {
        try {
            parameters.getString(paramName)
        } catch (Exception e) {
            System.err.println "Parameter $paramName not found, using default value: $defaultValue"
            defaultValue
        }
    }

    boolean getBoolean(String paramName, boolean defaultValue = false) {
        try {
            parameters.getBoolean(paramName)
        } catch (Exception e) {
            System.err.println "Parameter $paramName not found, using default value: $defaultValue"
            defaultValue
        }
    }
}
