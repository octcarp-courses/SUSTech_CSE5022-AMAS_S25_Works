package camera.agent

import camera.common.SpaceTrait
import camera.context.WorldManager
import camera.environment.TargetObject
import camera.utils.ParameterUtils
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
class Camera implements SpaceTrait {
    private final double CAMERA_RADIUS = ParameterUtils.instance.CAMERA_RADIUS
    private final double CAMERA_ANGLE = ParameterUtils.instance.CAMERA_ANGLE

    // managed target objects
    private List<TargetObject> ownedTarObjs = []

    double utility = 0.0

    Camera(ContinuousSpace space, int id) {
        this.space = space
        this.id = id
    }

    /**
     * The main execution loop of the camera.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    void step() {
        // update owned objects - hand over if necessary
        List<TargetObject> stillOwned = []
        ownedTarObjs.each { obj ->
            if (shouldHandOver(obj)) {
                handover(obj)
            } else {
                // no need to hand over - it will still be tracked by me if possible
                stillOwned << obj
            }
        }
        ownedTarObjs = stillOwned
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
    private boolean shouldHandOver(TargetObject tarObj) {
        return !isInFOV(tarObj)
    }

    // initiate handover
    private void handover(TargetObject tarObj) {
        // advertise owned objects to other cameras
        def world = WorldManager.instance
        def graph = world.visionGraph
        def neighbors = graph.getNeighbors(this.id)

        Map<Camera, Double> bids = [:]

        // receive bids (i.e., utility) from other cameras
        world.cameras.each { cam ->
            if (cam.id != this.id && neighbors.contains(cam.id)) {
                double bid = cam.getObjUtility(tarObj)
                if (bid > 0.0) {
                    bids[cam] = bid
                }
            }
        }

        if (bids.isEmpty()) return

            def sortedBids = bids.sort { -it.value }
        Camera winner = sortedBids.keySet().first()
        double highest = sortedBids[winner]
        double secondHighest = 0.0
        if (sortedBids.size() > 1) {
            secondHighest = sortedBids.values()[1]
        }

        // decide the winner and finalize transfer of object
        this.ownedTarObjs.remove(tarObj)
        winner.ownedTarObjs << tarObj

        // update the current utility of the buyer & seller cameras
        this.utility += secondHighest
        winner.utility -= secondHighest

        // update vision graph
        graph.reinforce(this.id, winner.id)
    }

    /**
     * Calculates the utility of an object if tracked by this camera.
     * 
     * NOTE: This is for calculating the bid for a specific object.
     * 
     * @param obj the object to be tracked
     * @return one single double value representing the utility
     */
    double getObjUtility(TargetObject tarObj) {
        def clarity = estimateClarity(tarObj)
        def visibility = estimateVisibility(tarObj)
        return clarity * visibility
    }

    /**
     * Calculates the current utility of the camera.
     * 
     * NOTE: This is for monitoring the instantaneous utility of the camera
     * 
     * @return one single double value representing the utility
     */
    double getUtility() {
        double totalUtility = 0.0
        ownedTarObjs.each { tarObj ->
            totalUtility += getObjUtility(tarObj)
        }
        return totalUtility
    }

    private double estimateClarity(TargetObject tarObj) {
        double distance = calcDxDyDistanceWithOther(tarObj)[2]

        return 1.0 - (distance / CAMERA_RADIUS);
    }

    private double estimateVisibility(TargetObject tarObj) {
        return isInFOV(tarObj) ? 1.0 : 0.0
    }

    private boolean isInFOV(TargetObject tarObj) {
        def res = calcDxDyDistanceWithOther(tarObj)

        double dx = res[0]
        double dy = res[1]
        double distance = res[2]

        if (distance > CAMERA_RADIUS) return false

        double angle = Math.toDegrees(Math.atan2(dy, dx))
        return Math.abs(angle) <= CAMERA_ANGLE / 2
    }
}

