package camera.common

import groovy.transform.CompileStatic

import repast.simphony.space.continuous.ContinuousSpace
import repast.simphony.space.continuous.NdPoint

import camera.utils.SpaceUtils

@CompileStatic
trait SpaceTrait {
    ContinuousSpace space
    int id

    boolean moveTo(double x, double y) {
        space.moveTo(this, x, y)
    }

    // Calculate (x difference, y difference, distance) with others (for FOV calculation)
    double[] calcDxDyDistanceWithOther(SpaceTrait other) {
        NdPoint thisLoc = space.getLocation(this)
        NdPoint otherLoc = space.getLocation(other)
        return SpaceUtils.calcDxDyDistance(thisLoc, otherLoc)
    }
}
