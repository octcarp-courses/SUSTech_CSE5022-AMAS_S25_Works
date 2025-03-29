package tileworld.context

import groovy.transform.CompileStatic

import repast.simphony.context.Context
import repast.simphony.context.space.graph.NetworkBuilder
import repast.simphony.context.space.grid.GridFactoryFinder
import repast.simphony.dataLoader.ContextBuilder
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.environment.RunListener
import repast.simphony.random.RandomHelper
import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridBuilderParameters
import repast.simphony.space.grid.GridPoint
import repast.simphony.space.grid.SimpleGridAdder
import repast.simphony.space.grid.WrapAroundBorders
import tileworld.agent.HoleStrategy
import tileworld.agent.Robot
import tileworld.data.DataHandler
import tileworld.utils.*

@CompileStatic
class TileWorldBuilder implements ContextBuilder {
    private Grid grid
    private Context context
    
    private int W
    private int H
    private int NUM_ROBOTS

    private double SIMULATION_TICKS
    private int SCHEDULE_TICK_DELAY

    private WorldManager worldManager
    private final List<Robot> robots = []

    private RunListener listener = null
    @Override
    Context build(Context context) {
        this.context = context
        
        context.id = "TileWorld"

        globalInitial()
        
        configRunEnv()

        grid = GridFactoryFinder.createGridFactory(null).createGrid(
                "grid", context,
                new GridBuilderParameters(
                    new WrapAroundBorders(), new SimpleGridAdder(), true, W, H
                )
                )

        addElements()
        
        return context
    }

    private void globalInitial() {
        def gridUtils = GridUtils.instance
        def params = ParameterUtils.instance
        def dataHandler = DataHandler.instance
        
        params.refresh()
        dataHandler.refresh()
        
        W = params.GRID_WIDTH
        H = params.GRID_HEIGHT
        NUM_ROBOTS = params.NUM_ROBOTS
        SIMULATION_TICKS = params.SIMULATION_TICKS

        if (params.RANDOM_SEED >= 0) {
            RandomHelper.setSeed(params.RANDOM_SEED)
        }
    }
    
    private void configRunEnv() {
        RunEnvironment.instance.endAt(SIMULATION_TICKS)
        
        if (!listener) {
            listener = new RunListener() {
                @Override
                void stopped() {
                    println "Simulation stopped, collecting data..."
                    DataHandler.instance.dealRobotsData(robots)
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

    private void addElements() {
        WorldManager.initialize(grid, context)
        worldManager = WorldManager.instance
        context.add(worldManager)
        worldManager.initailDynamicPart()

        new NetworkBuilder<>("path_network", context, true).buildNetwork()
        
        robots.clear()
        final int holeSId = ParameterUtils.instance.HOLE_STRATEGY
        (1..NUM_ROBOTS).each { i ->
            def (x, y) = [0, 0]
            while (true) {
                x = RandomHelper.nextIntFromTo(0, W - 1)
                y = RandomHelper.nextIntFromTo(0, H - 1)
                if (!grid.getObjectsAt(x, y).any()) {
                    break
                }
            }

            int strategyId = (holeSId > 0) ? holeSId : (i - 1) % HoleStrategy.values().length
            def robot = new Robot(grid, new GridPoint(x, y), "${i}", HoleStrategy.values()[strategyId])
            context << robot
            robot.place()
            robots << robot
        }
    }
}
