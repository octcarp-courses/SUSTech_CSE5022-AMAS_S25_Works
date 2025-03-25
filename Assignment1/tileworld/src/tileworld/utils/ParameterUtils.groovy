package tileworld.utils

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
    
    private final Parameters parameters = RunEnvironment.instance.parameters
    final int SYSTEM_RANDOM_SEED = getInt("randomSeed", -1)
    
    final int GRID_WIDTH = getInt("GRID_WIDTH", 20)
    final int GRID_HEIGHT = getInt("GRID_HEIGHT", 15)

    final int NUM_ROBOTS = getInt("NUM_ROBOTS", 5)
    final int NUM_TILES = getInt("NUM_TILES", 5)
    final int NUM_HOLES = getInt("NUM_HOLES", 5)
    final int NUM_OBSTACLES = getInt("NUM_OBSTACLES", 5)
    final int NUM_STATIONS = getInt("NUM_STATIONS", 5)

    final int SENSING_RADIUS = getInt("SENSING_RADIUS", 5)
    final int ENERGY_WARNING = getInt("ENERGY_WARNING", 30)

    final double SIMULATION_TICKS = getDouble("SIMULATION_TICKS", 1000.0)
    
    final int RANDOM_SEED = getInt("RANDOM_SEED", -1)

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
