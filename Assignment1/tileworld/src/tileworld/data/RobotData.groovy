package tileworld.data

import groovy.transform.CompileStatic
import groovy.transform.TupleConstructor

import repast.simphony.space.grid.GridPoint


@TupleConstructor
@CompileStatic
class RobotData {
    double tick
    String id
    int score
    int energy
    GridPoint location
    GridPoint target
}
