package tileworld.environment

import groovy.transform.CompileStatic

import repast.simphony.random.RandomHelper
import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import tileworld.common.GridTrait

@CompileStatic
class Hole implements GridTrait {
    final int score
    boolean claimed = false

    Hole(Grid grid, GridPoint location, int score = RandomHelper.nextIntFromTo(1, 15)) {
        this.grid = grid
        this.location = location
        this.score = score
    }

    String getLabel() { "$score" }

    @Override
    boolean equals(obj) {
        if (this.is(obj)) return true
        if (!(obj instanceof Hole)) return false
        def hole = obj as Hole
        location == hole.location
    }
}
