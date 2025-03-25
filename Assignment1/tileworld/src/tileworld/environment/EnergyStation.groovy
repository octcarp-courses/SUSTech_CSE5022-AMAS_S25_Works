package tileworld.environment

import groovy.transform.CompileStatic

import repast.simphony.space.grid.Grid
import repast.simphony.space.grid.GridPoint
import tileworld.common.GridTrait

@CompileStatic
class EnergyStation implements GridTrait {

    EnergyStation(Grid grid, GridPoint location) {
        this.grid = grid
        this.location = location
    }

    @Override
    boolean equals(obj) {
        if (this.is(obj)) return true
        if (!(obj instanceof EnergyStation)) return false
        def station = obj as EnergyStation
        location == station.location
    }
}

