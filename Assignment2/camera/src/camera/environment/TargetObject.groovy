package camera.environment

import camera.common.SpaceTrait

import groovy.transform.CompileStatic
import repast.simphony.engine.schedule.ScheduledMethod
import repast.simphony.random.RandomHelper
import repast.simphony.space.continuous.ContinuousSpace;

/***
 * A target object functions like a person moving in the environment. This
 * object should be tracked by a camera.
 */

@CompileStatic
class TargetObject implements SpaceTrait {
    boolean isTracked

    TargetObject(ContinuousSpace space, int id) {
        this.space = space
        this.id = id
        isTracked = false
    }

    /***
     * The main execution loop of the target object.
     */
    @ScheduledMethod(start = 1d, interval = 1d)
    void step() {
        // define the moving behavior of each target object
        randMove()
    }
    
    private void randMove() {
        def dx = RandomHelper.nextDoubleFromTo(-1, 1)
        def dy = RandomHelper.nextDoubleFromTo(-1, 1)
        space.moveByDisplacement(this, dx, dy)
    }
}
