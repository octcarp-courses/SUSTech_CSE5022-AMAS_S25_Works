package tileworld.agent

import groovy.transform.CompileStatic
import groovy.util.logging.Log4j
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.random.RandomHelper
import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint

import tileworld.common.GridTrait
import tileworld.context.WorldManager
import tileworld.data.RobotData
import tileworld.environment.*
import tileworld.utils.*


@Log4j
@CompileStatic
class Robot implements GridTrait {
    private final String id
    private HoleStrategy holeStrategy

    private final int X
    private final int Y

    private int energy = 100
    private int scoreEarned = 0
    private int stepFromLastHole = 0

    private boolean haveTile = false


    private final int SENSING_RADIUS = ParameterUtils.instance.SENSING_RADIUS
    private final int ENERGY_WARNING = ParameterUtils.instance.ENERGY_WARNING

    private Tile tileClaimed = null
    private Hole holeClaimed = null
    private EnergyStation nearestStation = null

    List<RobotData> robotDataList = []

    Robot(Grid grid, GridPoint location, String id, HoleStrategy holeStrategy) {
        this.grid = grid
        this.location = location
        this.id = id
        this.holeStrategy = holeStrategy

        X = grid.dimensions.width
        Y = grid.dimensions.height
    }

    @ScheduledMethod(start = 1d, interval = 1d)
    void step() {
        log.debug "======== AGENT LOOP ========"

        currentLocationOption()
        sensingActions()

        def target = determineMovementTarget()
        def targetLocation = determinTargetLocation(target)

        simpleMoveTowards(targetLocation)
        double tick = RunEnvironment.instance.currentSchedule.tickCount
        robotDataList << new RobotData(tick, id, scoreEarned, energy, location, targetLocation)

        log.debug "============================"
    }


    private GridPoint determinTargetLocation(GridTrait target) {
        if (target) {
            return target.location
        }
        if (energy > 0) {
            return new GridPoint(location.x + randomDirection(), location.y + randomDirection())
        }
        return location
    }

    private GridTrait determineMovementTarget() {
        nearestStation?.with {
            if (energy - 15 < calcPath(it)) {
                return it
            }
        }

        if (energy < ENERGY_WARNING) {
            return nearestStation
        }

        if (!haveTile && tileClaimed) {
            return tileClaimed
        }

        if (haveTile && holeClaimed) {
            return holeClaimed
        }

        return null
    }

    private void sensingActions() {
        log.debug "Looking for new goals..."

        def lookedTiles = GridUtils.instance.getObjectsWithRadius(this, Tile, SENSING_RADIUS)
        def filteredTiles = lookedTiles.findAll { !it.claimed }

        if (filteredTiles) {
            tileClaimed?.with { filteredTiles << it }
            def nearestTile = getClosestObject(filteredTiles)
            if (nearestTile != tileClaimed) {
                tileClaimed?.claimed = false
                nearestTile.claimed = true
                tileClaimed = nearestTile
            }
        }

        def lookedHoles = GridUtils.instance.getObjectsWithRadius(this, Hole, SENSING_RADIUS)
        def filteredHoles = lookedHoles.findAll { !it.claimed }

        if (filteredHoles) {
            holeClaimed?.with { filteredHoles << it }
            def bestHole = getBestHole(filteredHoles)
            if (bestHole != holeClaimed) {
                holeClaimed?.claimed = false
                bestHole.claimed = true
                holeClaimed = bestHole
            }
        }

        def lookedStations = GridUtils.instance.getObjectsWithRadius(this, EnergyStation, SENSING_RADIUS)
        if (lookedStations) {
            nearestStation = getClosestObject(lookedStations)
        }
    }

    private void simpleMoveTowards(GridTrait destObj) {
        simpleMoveTowards(destObj.location)
    }

    private void simpleMoveTowards(GridPoint destination) {
        int xDestination = destination.x
        int yDestination = destination.y
        log.debug "Moving towards: ($xDestination, $yDestination)"

        def dx = (xDestination <=> location.x)
        def dy = (yDestination <=> location.y)
        moveByDirection(dx, dy)
    }

    private void randomMove() {
        moveByDirection(randomDirection(), randomDirection())
    }

    private void moveByDirection(int dx, int dy) {
        if (energy <= 0 || (dx == 0 && dy == 0)) return

            def x = location.x
        def y = location.y

        if (dx == 0) {
            if (isBlockedMove(x, y + dy)) {
                dy = 0
                dx = randomDirection()
            }
        } else if (dy == 0) {
            if (isBlockedMove(x + dx, y)) {
                dx = 0
                dy = randomDirection()
            }
        } else {
            def xBlocked = isBlockedMove(x + dx, y)
            def yBlocked = isBlockedMove(x, y + dy)

            if (xBlocked && yBlocked) {
                dx = -dx
                dy = -dy
                if (RandomHelper.nextIntFromTo(0, 1) == 0) dx = 0 else dy = 0
            } else {
                if (xBlocked) dx = 0
                else if (yBlocked) dy = 0
                else if (RandomHelper.nextIntFromTo(0, 1) == 0) dx = 0 else dy = 0
            }
        }

        def nextX = x + dx
        def nextY = y + dy
        if (isBlockedMove(nextX, nextY)) return

            def nextLoc = new GridPoint(nextX, nextY)
        changeLocPlaceTo(nextLoc)

        --energy
        ++stepFromLastHole
    }

    private void currentLocationOption() {
        grid.getObjectsAt(location.x, location.y).each { obj ->
            switch (obj) {
                case EnergyStation:
                    energy = 100
                    break
                case Hole:
                    def hole = obj as Hole
                    if (haveTile && (!hole.claimed || hole == holeClaimed)) {
                        if (WorldManager.instance.removeHole(location)) {
                            hole.claimed = true
                            haveTile = false
                            scoreEarned += hole.score
                            holeClaimed = null
                            stepFromLastHole = 0
                        }
                    }
                    break
                case Tile:
                    def tile = obj as Tile
                    if (!haveTile && (!tile.claimed || tile == tileClaimed)) {
                        if (WorldManager.instance.removeTile(location)) {
                            tile.claimed = true
                            haveTile = true
                            tileClaimed = null
                            log.debug "Picked up a tile"
                        }
                    }
                    break
            }
        }
    }

    private def <T extends GridTrait> T getClosestObject(List<T> objects) {
        objects.min { calcPath(it) }
    }

    private Hole getBestHole(List<Hole> filteredHoles) {
        switch (holeStrategy) {
            case HoleStrategy.NEAREST:
                return getClosestObject(filteredHoles)
            case HoleStrategy.HIGHEST:
                return filteredHoles.max { it.score }
            case HoleStrategy.A_STAR:
                return filteredHoles.max { it.score / (calcPath(it) + stepFromLastHole) }
        }
    }

    private boolean isBlockedMove(int x, int y) {
        if (x < 0 || x >= X || y < 0 || y >= Y) return true

        grid.getObjectsAt(x, y).any { it instanceof Obstacle || it instanceof Robot }
    }

    private static int randomDirection() {
        RandomHelper.nextIntFromTo(0, 1) * 2 - 1
    }

    String getLabel() {
        "${id} (${energy}%)"
    }
}
