package tileworld.environment

import groovy.transform.CompileStatic

import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import tileworld.common.GridTrait

@CompileStatic
class Tile implements GridTrait {
    boolean claimed = false

    Tile(Grid grid, GridPoint location) {
        this.grid = grid
        this.location = location
    }

    @Override
    boolean equals(obj) {
        if (this.is(obj)) return true
        if (!(obj instanceof Tile)) return false
        def tile = obj as Tile
        return location == tile.location
    }
}
