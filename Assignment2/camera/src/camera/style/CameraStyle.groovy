package camera.style

import camera.agent.Camera
import groovy.transform.CompileStatic
import repast.simphony.visualizationOGL2D.DefaultStyleOGL2D
import repast.simphony.visualizationOGL2D.StyleOGL2D
import java.awt.Color
import java.awt.Font
import java.awt.Graphics2D
import java.awt.geom.Path2D

import saf.v3d.scene.VSpatial

import repast.simphony.space.continuous.ContinuousSpace

@CompileStatic
class CameraStyle extends DefaultStyleOGL2D {

//    @Override
//    VSpatial getVSpatial(Object agent, VSpatial spatial) {
//        if (agent instanceof Camera) {
//            Camera cam = agent as Camera
//            ContinuousSpace space = cam.space
//            def location = space.getLocation(cam)
//
//            double x = location.x
//            double y = location.y
//            double radius = cam.RADIUS
//            double angle = cam.ANGLE
//            double rotation = cam.ROTATION
//
//            double angle1 = Math.toRadians(rotation + angle / 2)
//            double angle2 = Math.toRadians(rotation - angle / 2)
//
//            double x1 = x + radius * Math.cos(angle1)
//            double y1 = y + radius * Math.sin(angle1)
//            double x2 = x + radius * Math.cos(angle2)
//            double y2 = y + radius * Math.sin(angle2)
//
//            Path2D.Double triangle = new Path2D.Double()
//            triangle.moveTo(x, y)
//            triangle.lineTo(x1, y1)
//            triangle.lineTo(x2, y2)
//            triangle.closePath()
//
//            return shapeFactory.createShape(triangle)
//        }
//    }

    @Override
    String getLabel(Object object) {
        def camera = object as Camera
        return camera.getLabel()
    }

    @Override
    Font getLabelFont(Object object) {
        new Font("Arial", Font.BOLD, 14);
    }

    @Override
    Color getColor(Object agent) {
        return new Color(0, 0, 0, 0)
    }

    @Override
    int getBorderSize(Object object) {
        return 2
    }

    @Override
    float getScale(Object object) {
        return 5.0f
    }
}