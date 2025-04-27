package camera.agent

import camera.common.SpaceTrait
import camera.context.WorldManager
import camera.environment.Target
import camera.graph.PheGraph
import camera.utils.ParameterUtils
import camera.common.BidRec
import camera.agent.CameraParam

import groovy.transform.CompileStatic

import repast.simphony.random.RandomHelper
import repast.simphony.context.Context
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.space.continuous.ContinuousSpace
import repast.simphony.space.continuous.NdPoint

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
    private final Context context

    private final double RADIUS
    private final double ANGLE
    private final double ROTATION
    private final int MAX_TRACK = ParameterUtils.instance.CAMERA_MAX_TRACK

    private final WorldManager world = WorldManager.instance
    private final PheGraph graph = world.visionGraph

    private double payment = 0.0
    private double pReceive = 0.0
    private double utility = 0.0


    // managed target objects
    private List<Target> ownedTargets = []

    private Map<Target, List<BidRec>> recivedBid = [:]
    private Map<Target, Double> ownedUtilities = [:]

    private int curStep = 0

    Camera(Context context, ContinuousSpace space, int id, double rotation) {
        this(context, space, id, rotation, ParameterUtils.instance.CAMERA_RADIUS, ParameterUtils.instance.CAMERA_ANGLE)
    }

    Camera(Context context, ContinuousSpace space, int id, double rotation, double radius, double angle) {
        this.context = context
        this.space = space
        this.id = id
        this.RADIUS = radius
        this.ANGLE = angle
        this.ROTATION = rotation
    }

    /**
     * The main execution loop of the camera.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    void step() {
        ++curStep
        updateInfo()
        // update owned objects - hand over if necessary
        List<Target> stillOwned = []
        List<Target> needHandover = []
        ownedTargets.each { target ->
            if (shouldHandOver(target)) {
                boolean handRes = handOver(target)
                target.loseTrackBy(id)
            } else {
                stillOwned << target
            }
        }
        ownedTargets = stillOwned
        //		needHandover.each { target ->
        //			handover(target)
        //		}
        // track owned objects
        trackObjects()
    }

    private void updateInfo() {
        double totalUtility = 0.0
        payment = 0.0
        pReceive = 0.0
        ownedUtilities.clear()
        ownedTargets.each { target ->
            double utility = getTargetUtility(target)
            ownedUtilities[target] = utility
            totalUtility += utility
        }
        ownedUtilities.sort { a, b -> a.value <=> b.value }
    }

    /**
     * Simulate the behavior of object tracking
     */
    private void trackObjects() {
        // with limited resources, sometimes I can only track some objects
        int spare = MAX_TRACK - ownedTargets.size()

        // if no spare
        if (spare == 0) return

            int newCount = 0
        // Get targets
        def newTargets = getAvailableTargets()
        // Sort by there utility
        newTargets.sort{ -getTargetUtility(it) }

        // While loop
        int newTargetI = 0
        while(spare > newCount && newTargetI < newTargets.size()) {
            // Get the target
            def target = newTargets[newTargetI]
            // Double check not tracked by other camera
            if (!target.isTracked) {
                // Track it
                target.trackByCamera(id)
                // Add target to owned
                ownedTargets << target
                ++newCount
            }
            ++newTargetI
        }
    }

    // check whether the camera should initiate a handover
    private boolean shouldHandOver(Target target) {
        return (ownedUtilities[target] <=0.0)
    }

    /**
     * Hand over method using Vickrey Auction
     * 
     * @param target target need to hand over
     * @return Boolean value, whether hand over success
     */
    private boolean handOver(Target target) {
        recivedBid[target] = []
        int targetId = target.id

        // advertise owned objects to other cameras
        // Get probabilities from vision graph
        def neiProbabilities = graph.getNotifyProbabilities(id)

        // For each neighbor
        neiProbabilities.each { camId, probability ->
            if (probability >= 1 || RandomHelper.nextDouble() > probability) {
                // Send the neighbor, call its receive method for simulation
                sendTo(camId).receiveAuction(this.id, targetId)
            }
        }

        // receive bids (i.e., utility) from other cameras
        def bids = recivedBid[target]

        // No response
        if (bids.isEmpty()) {
            return false
        }

        // Sort the bid
        def sortedBids = bids.sort { -it.bid }

        // Decide the winner
        def winnerBid = sortedBids.first()

        // Decide the final bid
        double finalBid = 0.0
        // Have second bidder
        if (sortedBids.size() >= 2) {
            finalBid = sortedBids[1].bid
        }

        // decide the winner and finalize transfer of object
        // update the current utility of the buyer & seller cameras
        double thisUtility = ownedUtilities[target]
        // Can't hand over for utility is not enough
        if (thisUtility > 0 && finalBid <= thisUtility) {
            return false
        }

        // Get winner ID
        def winnerId = winnerBid.bidderId

        // Auctioneer sent
        sendTransferedTarget(target, finalBid)
        // Winner receive
        sendTo(winnerId).receiveTransferedTarget(target, finalBid)

        // update vision graph for success trade
        graph.reinforce(this.id, winnerId)

        return true
    }

    /**     
     * Calculates own bid for specific target from .
     * 
     * @param auctioneer Auctioneer who send the request
     * @param target The target object
     */
    void receiveAuction(int auctioneerId, int targetId) {
        // Get the utility
        double bid = getTargetUtility(world.getTargetById(targetId))
        // Judge if can handle the new one
        if (ownedTargets.size() < MAX_TRACK && bid > 0) {
            // Create record
            def bidRec = new BidRec(id, auctioneerId, targetId, bid)
            // Send record
            sendTo(auctioneerId).receiveBid(bidRec)
        }
    }

    /**
     * Receives and processes a bid record.
     * 
     * @param bidRec The bid record containing auctioneer ID, bid amount, and target ID.
     */
    void receiveBid(BidRec bidRec) {
        // Wrong
        if (bidRec.auctioneerId != this.id) return
            // Invalid
            if (bidRec.bid <= 0) return

            // Have target
            def target = recivedBid.keySet().find { it.id == bidRec.targetId }
        // Collect bid record
        if (target) {
            recivedBid[target] << bidRec
        }
    }

    /**
     * Winner bidder receive the target object
     * 
     * @param target Received target
     * @param bid final bid
     */
    void receiveTransferedTarget(Target target, double bid) {
        // Add to owned
        ownedTargets << target
        // Track the camera
        target.trackByCamera(id)
        // Payment increase
        payment += bid
    }

    /**
     * Auctioneer send the target object
     * 
     * @param target Sent target
     * @param bid Auctioneer
     */
    private void sendTransferedTarget(Target target, double bid) {
        // Lose the target track
        target.loseTrackBy(id)
        // Received payment increase
        pReceive += bid
    }

    /**
     * For simulate communication
     *
     * @param cameraId send to camera's id
     * @return the camera object reference
     */
    private Camera sendTo(int cameraId) {
        world.getCameraById(cameraId)
    }

    /**
     * Calculates the utility of an object if tracked by this camera.
     * 
     * NOTE: This is for calculating the bid for a specific object.
     * 
     * @param target the object to be tracked
     * @return one single double value representing the utility
     */
    double getTargetUtility(Target target) {
        // Similar steps like FOV calculation
        def res = calcDxDyDistanceWithOther(target)
        double dx = -res[0]
        double dy = -res[1]
        double distance = res[2]
        double angle = Math.toDegrees(Math.atan2(dy, dx))
        double relativeAngle = ROTATION - angle

        // Calculate angle factor
        double factor = Math.abs(relativeAngle) / (ANGLE / 2)
        double angleVis = 1 / (1 + factor) - 0.5
        // Calculate radius factor
        double radiusVis = 1.0 - (distance / RADIUS)
        // Get v
        double visibility = angleVis * radiusVis

        // Calculate confidence
        double confidence = isInFOV(target) ? 1.0 : 0.0

        return confidence * visibility
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
        ownedTargets.each { target ->
            totalUtility += getTargetUtility(target)
        }
        return totalUtility
    }

    /**
     * Judge whether the target is in this camera FOV.
     * 
     * @param target Input target object
     * @return boolean value whether the target is in FOV
     */
    private boolean isInFOV(Target target) {
        // Get (x difference, y difference, distance) from other util methods
        def res = calcDxDyDistanceWithOther(target)

        // Get value for FOV calculation
        double dx = -res[0]
        double dy = -res[1]
        double distance = res[2]

        // Outside radius
        if (distance > RADIUS) {
            return false
        }

        // Calculate the absolute angle of an object (-180, 180)
        double angle = Math.toDegrees(Math.atan2(dy, dx))

        // Get relative angle
        double relativeAngle = ROTATION - angle

        // Determine whether it is within the angle range
        return Math.abs(relativeAngle) <= ANGLE / 2
    }

    private List<Target> getAvailableTargets() {
        List<Target> availableTargets = []
        context.getObjects(Target)
                .findAll { obj ->
                    def target = obj as Target
                    if (!target.isTracked && isInFOV(target)) {
                        availableTargets << target
                    }
                }
        availableTargets
    }

    String getLabel() {
        return "c$id"
    }
}

