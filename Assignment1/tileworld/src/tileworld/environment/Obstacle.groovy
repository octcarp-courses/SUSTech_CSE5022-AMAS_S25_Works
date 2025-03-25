package tileworld.environment

import groovy.transform.CompileStatic

import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import tileworld.common.GridTrait

@CompileStatic
class Obstacle implements GridTrait {

    Obstacle(Grid grid, GridPoint location) {
        this.grid = grid
        this.location = location
    }
}
