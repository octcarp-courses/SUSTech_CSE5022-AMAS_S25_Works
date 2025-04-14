package tileworld.context

import groovy.transform.CompileStatic
import groovy.util.logging.Log4j

import repast.simphony.context.Context
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.random.RandomHelper
import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import repast.simphony.util.ContextUtils

import tileworld.utils.GridUtils
import tileworld.utils.ParameterUtils
import tileworld.utils.StationPlacer
import tileworld.agent.Robot
import tileworld.common.GridTrait
import tileworld.environment.EnergyStation
import tileworld.environment.Hole
import tileworld.environment.Obstacle
import tileworld.environment.Tile

@CompileStatic
class WorldManager {
    private static WorldManager instance = null

    private final Grid grid
    private final Context context
    private final int X
    private final int Y

    private WorldManager(Grid grid, Context context) {
        this.grid = grid
        this.context = context
        X = grid.dimensions.width
        Y = grid.dimensions.height
    }

    static synchronized void initialize(Grid grid, Context context) {
        instance = new WorldManager(grid, context)
    }

    static WorldManager getInstance() {
        if (!instance) {
            System.err.println "Forget to initailize World Manager, check your code."
        }
        instance
    }

    private final int NUM_TILES = ParameterUtils.instance.NUM_TILES
    private final int NUM_HOLES = ParameterUtils.instance.NUM_HOLES
    private final int NUM_OBSTACLES = ParameterUtils.instance.NUM_OBSTACLES
    private final int NUM_STATIONS = ParameterUtils.instance.NUM_STATIONS

    private final List<Tile> tiles = []
    private final List<Hole> holes = []
    private final List<Obstacle> obstacles = []
    private final List<EnergyStation> stations = []


    @ScheduledMethod(start = 50d, interval = 100d)
    void reconfigureElements() {
        obstacles.each { it.changeLocPlaceTo(randomValidPoint()) }
        tiles.each { it.changeLocPlaceTo(randomValidPoint()) }
        holes.each { it.changeLocPlaceTo(randomValidPoint()) }
    }

    void initailDynamicPart() {
        List<GridPoint> stationPoints = StationPlacer.getResult(NUM_STATIONS, X as double, Y as double)

        stationPoints.each { stationP ->
            def station = new EnergyStation(grid, stationP)
            context << station
            station.place()
            stations << station
        }

        NUM_OBSTACLES.times {
            def obstacle = new Obstacle(grid, randomValidPoint())
            context << obstacle
            obstacle.place()
            obstacles << obstacle
        }

        NUM_TILES.times { addOneTile() }
        NUM_HOLES.times { addOneHole() }
    }

    boolean removeTile(GridPoint removeLoc) {
        def tileToRemove = tiles.find { it.location == removeLoc }

        if (tileToRemove) {
            tiles.remove(tileToRemove)
            context.remove(tileToRemove)
            addOneTile()
            return true
        }
        false
    }

    boolean removeHole(GridPoint removeLoc) {
        def holeToRemove = holes.find { it.location == removeLoc }

        if (holeToRemove) {
            holes.remove(holeToRemove)
            context.remove(holeToRemove)
            addOneHole()
            return true
        }
        false
    }

    private GridPoint randomValidPoint() {
        int x, y;
        while (true) {
            x = RandomHelper.nextIntFromTo(0, X - 1)
            y = RandomHelper.nextIntFromTo(0, Y - 1)
            if (!grid.getObjectsAt(x, y).any()) break
        }

        new GridPoint(x, y)
    }

    private void addOneHole() {
        def hole = new Hole(grid, randomValidPoint())
        context.add(hole)
        hole.place()
        holes << hole
    }

    private void addOneTile() {
        def tile = new Tile(grid, randomValidPoint())
        context << tile
        tile.place()
        tiles << tile
    }
}
