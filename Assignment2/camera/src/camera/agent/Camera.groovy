package camera.agent

import camera.environment.TargetObject

import groovy.transform.CompileStatic

import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.space.continuous.ContinuousSpace

/**
 * A camera takes responsibility of tracking one or more target objects in the
 * environment.
 * 
 * It is initialized in a fixed location and cannot move, but it can rotate a
 * certain angle (we do not need to implement this level of complexity). Simply
 * put, its Field of View (FOV) is represented as a triangle in 2D space.
 * 
 * Each camera uses a shared model to calculate its utility - capability of
 * tracking the target objects.
 * 
 * When a target object moves out of the camera's FOV, the camera will start a
 * Vickrey Auction to attempt transfer of this target object to other neighbor
 * cameras.
 * 
 * The neighborhood of cameras are maintained using a vision graph. This vision
 * graph can be dynamically updated using an approach similar to Ant Colony
 * Optimization (ACO).
 */
@CompileStatic
class Camera {
    ContinuousSpace<Object> space
    int id
    // managed target objects
    List<TargetObject> ownedObjs = []

    Camera(ContinuousSpace<Object> space, int id) {
        this.space = space
        this.id = id
    }

    /**
     * The main execution loop of the camera.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    def step() {
        // update owned objects - hand over if necessary
        List<TargetObject> stillOwned = []
        ownedObjs.each { obj ->
            if (shouldHandOver(obj)) {
                handover(obj)
            } else {
                // no need to hand over - it will still be tracked by me if possible
                stillOwned << obj
            }
        }
        ownedObjs = stillOwned
        // track owned objects
        trackObjects()
    }

    // simulate the behavior of object tracking
    private void trackObjects() {
        // TODO: with limited resources, sometimes I can only track some objects
        // TODO: also define some logic so we can collect the performance of object
        // tracking (how long an object is tracked/missed)
    }

    // check whether the camera should initiate a handover
    private boolean shouldHandOver(TargetObject obj) {
        // TODO
        return false
    }

    // initiate handover
    private void handover(TargetObject obj) {
        // TODO: advertise owned objects to other cameras
        // TODO: receive bids (i.e., utility) from other cameras
        // TODO: decide the winner and finalize transfer of object
        // TODO: update the current utility of the buyer & seller cameras
        // TODO: update vision graph
    }

    /**
     * Calculates the utility of an object if tracked by this camera.
     * 
     * NOTE: This is for calculating the bid for a specific object.
     * 
     * @param obj the object to be tracked
     * @return one single double value representing the utility
     */
    def getObjUtility(TargetObject obj) {
        // TODO
        return -1
    }

    /**
     * Calculates the current utility of the camera.
     * 
     * NOTE: This is for monitoring the instantaneous utility of the camera
     * 
     * @return one single double value representing the utility
     */
    def getUtility() {
        // TODO
        return -1
    }

    int getId() {
        return id
    }
}

