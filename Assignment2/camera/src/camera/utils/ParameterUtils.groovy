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
    
    final int SYSTEM_RANDOM_SEED = getInt("randomSeed", -1)
    
    final int SCENARIO_ID = getInt("SCENARIO_ID", 0)

    final double CAMERA_RADIUS = getDouble("CAMERA_RADIUS", 15.0)
    final double CAMERA_ANGLE = getDouble("CAMERA_ANGLE", 120.0)
    final int CAMERA_MAX_TRACK = getInt("CAMERA_MAX_TRACK", 5)

    // Number of target objects in the simulation
    final int NUM_TARGETS = getInt("NUM_TARGETS", 10)

    // Pheromone Rho, evaporation rate
    final double PHEROMONE_RHO = getDouble("PHEROMONE_RHO", 0.1)
    // Pheromone Delta, trade increases
    final double PHEROMONE_DELTA = getDouble("PHEROMONE_DELTA", 1.0)

    // Probability Epsilon, for weak threshold
    final double PROBABILITY_EPS = getDouble("PROBABILITY_EPS", 0.1)
    // probability Eta, used to notify weak neighbors in the simulation
    final double PROBABILITY_ETA = getDouble("PROBABILITY_ETA", 0.1)

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
