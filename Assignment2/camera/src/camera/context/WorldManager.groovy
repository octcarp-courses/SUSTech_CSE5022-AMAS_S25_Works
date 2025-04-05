package camera.context

import camera.agent.Camera
import camera.environment.TargetObject
import groovy.transform.CompileStatic
import repast.simphony.context.Context
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.space.continuous.ContinuousSpace

/***
 * The world controller is responsible for initializing the agents and
 * maintaining the vision graph.
 */

@CompileStatic
class WorldManager {
    // world
    private ContinuousSpace space
    // cameras
    private List<Camera> cameras
    // target objects
    private int numTargetObjects
    private List<TargetObject> targetObjects

    WorldManager(ContinuousSpace space, int numTargetObjects) {
        this.space = space
        this.numTargetObjects = numTargetObjects
    }

    /***
     * Initialize Repast Simphony agents like target objects and cameras.
     * 
     * @param context Repast Simphony's current simulation context
     */
    void initAgents(Context context) {
        this.targetObjects = initTargetObjects()
        this.cameras = initCameras()
    }

    private List<TargetObject> initTargetObjects() {
        // TODO
        return []
    }

    private List<Camera> initCameras() {
        // TODO
        return []
    }

    /***
     * Handles pheromone levels of edges in the vision graph.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    void handlePheromone() {
        // TODO: evaporate pheromone
    }
}