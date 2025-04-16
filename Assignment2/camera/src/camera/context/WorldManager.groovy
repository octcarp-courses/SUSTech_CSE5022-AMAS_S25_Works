package camera.context

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.space.continuous.ContinuousSpace

import camera.agent.Camera
import camera.environment.TargetObject
import camera.environment.VisionGraph
import camera.utils.ParameterUtils
import camera.context.WorldManager

/***
 * The world controller is responsible for initializing the agents and
 * maintaining the vision graph.
 */

@CompileStatic
class WorldManager {

    private static WorldManager instance = null

    static synchronized void initInstance(Context context, ContinuousSpace space) {
        instance = new WorldManager(context, space)
    }

    static WorldManager getInstance() {
        if (!instance) {
            System.err.println "Forget to initailize World Manager, check your code."
        }
        instance
    }

    // context
    private final Context context
    // world
    private final ContinuousSpace space

    private final int NUM_CAMERAS = ParameterUtils.instance.NUM_CAMERAS
    private final int NUM_TARGET_OBJS = ParameterUtils.instance.NUM_TARGET_OBJS

    private int globalId = 0

    // cameras
    private final List<Camera> cameras = []
    // target objects
    private final List<TargetObject> targetObjects = []

    private final VisionGraph visionGraph = new VisionGraph()

    private WorldManager(Context context, ContinuousSpace space) {
        this.context = context
        this.space = space
    }

    /***
     * Initialize Repast Simphony agents like target objects and cameras.
     * 
     * @param context Repast Simphony's current simulation context
     */
    void initWorld() {
        globalId = 0

        NUM_CAMERAS.times {
            def camera = new Camera(space, ++globalId)
            cameras << camera
            context << camera
        }

        NUM_TARGET_OBJS.times {
            def targetObject = new TargetObject(space, ++globalId)
            targetObjects << targetObject
            context << targetObject
        }
    }

    /***
     * Handles pheromone levels of edges in the vision graph.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    void handlePheromone() {
        // evaporate pheromone
        visionGraph.evaporate()
    }
}