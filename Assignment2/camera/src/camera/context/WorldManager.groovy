package camera.context

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.space.continuous.ContinuousSpace

import camera.agent.Camera
import camera.environment.Target
import camera.graph.PheGraph
import camera.utils.ParameterUtils
import camera.context.WorldManager

/***
 * The world manager is responsible for initializing the agents and
 * maintaining the vision graph.
 */
@CompileStatic
class WorldManager {

    private static WorldManager instance = null

    static synchronized void initInstance(Context context, ContinuousSpace space, CameraScenario scenario) {
        instance = new WorldManager(context, space, scenario)
    }

    static WorldManager getInstance() {
        if (!instance) {
            System.err.println "Forget to initailize World Manager, check your code."
        }
        instance
    }

    final List<Integer> trackedCount = []
    final List<Map<Integer, Map<Integer, Double>>> graphSnapshots = []

    // context
    private final Context context
    // world
    private final ContinuousSpace space

    private final int NUM_TARGET_OBJS = ParameterUtils.instance.NUM_TARGETS

    final PheGraph visionGraph
    // cameras
    private final Map<Integer, Camera> cameras = [:]
    // target objects
    private final Map<Integer, Target> targets = [:]

    private final CameraScenario scenario

    private WorldManager(Context context, ContinuousSpace space, CameraScenario scenario) {
        this.context = context
        this.space = space
        this.scenario = scenario
        visionGraph = new PheGraph(scenario.cameraParams.size())
    }

    /***
     * Initialize Repast Simphony agents like target objects and cameras.
     * 
     * @param context Repast Simphony's current simulation context
     */
    void initWorld() {
        int cameraId = 0
        scenario.cameraParams.each { param ->
            int id = ++cameraId
            def camera = new Camera(context, space, id, param.rotation)
            cameras[cameraId] = camera
            context << camera
            camera.moveTo(param.x, param.y)
        }

        (1..NUM_TARGET_OBJS).each { id ->
            def target = new Target(space, id)
            targets[id] = target
            context << target
        }
    }

    int tick = 0
    @ScheduledMethod(start = 2d, interval = 1d)
    void step() {
        if (tick % 300 == 0) {
            graphSnapshots << visionGraph.graphSnapshot()
        }
        trackedCount << (targets.values().count { it.isTracked } as int)
        handlePheromone()
        ++tick
    }

    /***
     * Handles pheromone levels of edges in the vision graph.
     */
    void handlePheromone() {
        // evaporate pheromone
        visionGraph.evaporateLastStep()
        visionGraph.initThisStep()
    }

    Camera getCameraById(int id) {
        def camera = cameras[id]
        if (!camera) {
            System.err.println "Camera $id not found."
        }
        camera
    }

    Target getTargetById(int id) {
        def target = targets[id]
        if (!target) {
            System.err.println "Target object $id not found."
        }
        target
    }
}