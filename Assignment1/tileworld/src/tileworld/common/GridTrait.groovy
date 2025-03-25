package tileworld.common

import groovy.transform.CompileStatic

import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint

@CompileStatic
trait GridTrait {
    Grid grid
    GridPoint location

    void changeLocPlaceTo(int x, int y) {
        changeLocPlaceTo(new GridPoint(x, y))
    }

    void changeLocPlaceTo(GridPoint destination) {
        location = destination
        place()
    }

    void place() {
        grid.moveTo(this, location.x, location.y)
    }

    int getPathSteps(GridPoint otherLocation) {
        Math.abs(location.x - otherLocation.x) + Math.abs(location.y - otherLocation.y)
    }

    int getPathSteps(GridTrait other) {
        getPathSteps(other.location)
    }
    
    int calcPath(GridPoint other) {
        Math.abs(location.x - other.x) + Math.abs(location.y - other.y)
    }

    int calcPath(GridTrait other) {
        calcPath(other.location)
    }
}
