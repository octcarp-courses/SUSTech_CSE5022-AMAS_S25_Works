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
    
    private final String FILE_DIR = './result/temp/'
    
    private int times = 0
    
    private String fileName = 'default'
    
    private DataHandler() {
        initDirectory()
        initFileName()
    }
    
    void refresh() {
        initFileName()
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
    
    private void initFileName() {
        def param = ParameterUtils.instance
        def W = "W_${param.GRID_WIDTH}"
        def H = "H_${param.GRID_HEIGHT}"
        def RO = "RO_${param.NUM_ROBOTS}"
        def TI = "TI_${param.NUM_TILES}"
        def HO = "HO_${param.NUM_HOLES}"
        def OB = "OB_${param.NUM_OBSTACLES}"
        def ST = "ST_${param.NUM_STATIONS}"
        def SR = "SR_${param.SENSING_RADIUS}"
        def HOST = "HO-ST_${param.HOLE_STRATEGY}"
        def EW = "EW_${param.ENERGY_WARNING}"
        def seed = "seed_${param.SYSTEM_RANDOM_SEED}"
        
        fileName = "${W}_${H}_${RO}_${TI}_${HO}_${OB}_${ST}_${SR}_${HOST}_${EW}_${seed}"
    }

    
    void dealRobotsData(List<Robot> robots) {
        List<RobotData> allRobotData = []
        robots.each { robot ->
            allRobotData.addAll(robot.robotDataList)
        }
        
        def filePath = "${FILE_DIR}${fileName}.csv"
        CSVWriter writer = new CSVWriter(new FileWriter(filePath, false))
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

