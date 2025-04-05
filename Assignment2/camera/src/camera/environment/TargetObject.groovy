package camera.environment

import groovy.transform.CompileStatic
import repast.simphony.space.continuous.ContinuousSpace;

/***
 * A target object functions like a person moving in the environment. This
 * object should be tracked by a camera.
 */

@CompileStatic
class TargetObject {
    ContinuousSpace space
    int id
    boolean isTracked

    TargetObject(ContinuousSpace space, int id) {
        this.space = space
        this.id = id
        this.isTracked = false
    }

    /***
     * The main execution loop of the target object.
     */
    void step() {
        // TODO: define the moving behavior of each target object
    }
}


