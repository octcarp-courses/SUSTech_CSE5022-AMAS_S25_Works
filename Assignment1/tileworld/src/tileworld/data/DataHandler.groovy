package tileworld.data

import au.com.bytecode.opencsv.CSVWriter
import groovy.transform.CompileStatic
import tileworld.agent.Robot
import tileworld.utils.GridUtils
import tileworld.utils.ParameterUtils
import repast.simphony.data2.AggregateDataSource
import repast.simphony.data2.DataSet
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.schedule.ScheduledMethod

@CompileStatic
class DataHandler {
    private static DataHandler instance = null
    
    private final String FILE_DIR = "./result/temp/"
    
    private DataHandler() {
        initDirectory()
    }

    static DataHandler getInstance() {
        if (!instance) {
            synchronized (DataHandler) {
                if (!instance) instance = new DataHandler()
            }
        }
        instance
    }
        
    private void initDirectory() {
        File dir = new File(FILE_DIR)
        if (!dir.exists()) {
            dir.mkdirs()
            println "Creat Directory: ${dir.absolutePath}"
        } else {
            println "Directory existed: ${dir.absolutePath}"
        }
    }

    
    void dealRobotsData(List<Robot> robots) {
        List<RobotData> allRobotData = []
        robots.each { robot ->
            allRobotData.addAll(robot.robotDataList)
        }
        
        def filePath = "${FILE_DIR}robot_data_seed_${ParameterUtils.instance.SYSTEM_RANDOM_SEED}.csv"
        CSVWriter writer = new CSVWriter(new FileWriter(filePath))
        writer.writeNext(["tick", "robot_id", "score", "energy", "cur_loc", "tar_loc"] as String[])
    
        allRobotData.each { record ->
            writer.writeNext([
                record.tick.toString(),
                record.id.toString(),
                record.score.toString(),
                record.energy.toString(),
                record.location.toString(),
                record.target.toString()
            ] as String[])
        }
        
        writer.close()
        
        println "Successfully write to ${filePath}"
    }
}

