package camera.context

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.context.space.continuous.ContinuousSpaceFactoryFinder
import repast.simphony.context.space.graph.NetworkBuilder;
import repast.simphony.dataLoader.ContextBuilder
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.environment.RunListener
import repast.simphony.space.continuous.ContinuousSpace
import repast.simphony.space.continuous.RandomCartesianAdder
import repast.simphony.space.continuous.WrapAroundBorders
import repast.simphony.space.grid.Grid
import camera.data.DataHandler
import camera.utils.ParameterUtils

@CompileStatic
class MultiCameraTrackingBuilder implements ContextBuilder {
    private static final String SPACE_NAME = "space"
    // network
    private static final String NET_NAME = "tracking network"

    private Context context
    private ContinuousSpace space

    private double SPACE_X_SIZE
    private double SPACE_Y_SIZE
    private int SCENARIO_ID

    private CameraScenario scenario

    private WorldManager worldManager

    private RunListener listener = null

    @Override
    Context build(Context context) {
        this.context = context
        context.setId("camera")

        // global initialize & add input parameters
        globalInitial()

        // configure run environment
        configRunEnv()

        scenario = CameraScenario.scenarios[SCENARIO_ID]

        // create continuous 2D space projection (randomize initialize location for new
        // objects - then manually move cameras into positions)
        def spaceFactory = ContinuousSpaceFactoryFinder.createContinuousSpaceFactory(null)
        space = spaceFactory.createContinuousSpace(SPACE_NAME, context,
                new RandomCartesianAdder(), new WrapAroundBorders(),
                scenario.worldX, scenario.worldY)

        // create tracking network as a projection for visualization
        // when a camera is actually tracking a target object, an edge will be formed
        def netBuilder = new NetworkBuilder(NET_NAME, context, true)
        netBuilder.buildNetwork()

        addElements()

        return context
    }

    private void globalInitial() {
        def params = ParameterUtils.instance

        params.refresh()

        SCENARIO_ID = params.SCENARIO_ID
    }

    private void addElements() {
        // create world controller for environment initialization and updating
        WorldManager.initInstance(context, space, scenario)

        worldManager = WorldManager.instance
        worldManager.initWorld()

        context << worldManager
    }

    private void configRunEnv() {
        RunEnvironment.instance.endAt(1000.0)

        if (!listener) {
            listener =new RunListener() {
                        @Override
                        void stopped() {
                            println "Simulation stopped, collecting data..."
                            DataHandler.instance.dealTrackCountData(WorldManager.instance.trackedCount)
                            DataHandler.instance.dealVisionGraph(WorldManager.instance.graphSnapshots)
                        }

                        @Override
                        void started() {
                            println "Simulation started"
                        }

                        @Override
                        void restarted() {
                            println "Simulation restarted"
                        }
                        @Override
                        void paused() {
                            println "Simulation paused"
                        }
                    }
            RunEnvironment.instance.addRunListener(listener)
        }
    }
}
