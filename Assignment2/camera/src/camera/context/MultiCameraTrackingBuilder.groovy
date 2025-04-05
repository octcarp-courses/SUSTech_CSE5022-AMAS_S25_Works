package camera.context

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.context.space.continuous.ContinuousSpaceFactoryFinder
import repast.simphony.context.space.graph.NetworkBuilder;
import repast.simphony.dataLoader.ContextBuilder
import repast.simphony.engine.environment.RunListener
import repast.simphony.space.continuous.RandomCartesianAdder
import repast.simphony.space.grid.Grid


@CompileStatic
class MultiCameraTrackingBuilder implements ContextBuilder {
    static final String SPACE_NAME = "space"
    static final double SPACE_X_SIZE = 50
    static final double SPACE_Y_SIZE = 50
    // network
    static final String NET_NAME = "tracking network"
    // target object
    static final int NUM_TARGET_OBJS = 10

    @Override
    Context build(Context context) {
        context.setId("camera")

        // TODO: add input parameters

        // create continuous 2D space projection (randomize initialize location for new
        // objects - then manually move cameras into positions)
        def spaceFactory = ContinuousSpaceFactoryFinder.createContinuousSpaceFactory(null)
        def space = spaceFactory.createContinuousSpace(SPACE_NAME, context,
                new RandomCartesianAdder<Object>(), new repast.simphony.space.continuous.WrapAroundBorders(),
                SPACE_X_SIZE, SPACE_Y_SIZE)

        // create tracking network as a projection for visualization
        // when a camera is actually tracking a target object, an edge will be formed
        def netBuilder = new NetworkBuilder(NET_NAME, context, true)
        netBuilder.buildNetwork()

        // create world controller for environment initialization and updating
        def worldCtrl = new WorldManager(space, NUM_TARGET_OBJS)
        context.add(worldCtrl)
        worldCtrl.initAgents(context)

        return context
    }
}
