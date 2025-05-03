package camera.data

import groovy.transform.CompileStatic

import au.com.bytecode.opencsv.CSVWriter

import repast.simphony.data2.AggregateDataSource
import repast.simphony.data2.DataSet
import repast.simphony.engine.environment.RunEnvironment
import repast.simphony.engine.schedule.ScheduledMethod

import camera.utils.ParameterUtils

/**
 * Class responsible for collect data
 */
@CompileStatic
class DataHandler {
    private static DataHandler instance = null

    private static final String FILE_DIR = './output/temp_csv/'

    private static final String trackCountFileName = 'track_count'
    private static final String pheGraphFileName = 'phe_graph'

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
        }

        println "Result .csv output Directory existed: ${dir.absolutePath}"
    }

    void dealTrackCountData(List<Integer> trackCount) {
        def filePath = "${FILE_DIR}${trackCountFileName}.csv"
        CSVWriter writer = new CSVWriter(new FileWriter(filePath, false))
        writer.writeNext(["tick", "track_count"] as String[])

        int tick = 1
        trackCount.each { count ->
            writer.writeNext([
                ++tick,
                count,
            ] as String[])
        }

        writer.close()

        println "Successfully write to ${filePath}"
    }

    void dealVisionGraph(List<Map<Integer, Map<Integer, Double>>> snapshots) {
        def filePath = "${FILE_DIR}${pheGraphFileName}.csv"
        CSVWriter writer = new CSVWriter(new FileWriter(filePath, false))
        writer.writeNext([
            "batch",
            "fromId",
            "toId",
            "value"
        ] as String[])

        int batch = 0
        snapshots.each { graph ->
            graph.each { from, toMap ->
                toMap.each { to, value ->
                    writer.writeNext([
                        batch,
                        from,
                        to,
                        value,
                    ] as String[])
                }
            }
            ++batch
        }

        writer.close()

        println "Successfully write to ${filePath}"
    }

    private void initFileName() {
        def param = ParameterUtils.instance
        def SC = "SC_${param.SCENARIO_ID}"
        def CR = "CR_${param.CAMERA_RADIUS}"
        def CA = "CA_${param.CAMERA_ANGLE}"
        def TM = "TM_${param.CAMERA_MAX_TRACK}"
        def TN = "TN_${param.NUM_TARGETS}"
        def PR = "PR_${param.PHEROMONE_RHO}"
        def PD = "PD_${param.PHEROMONE_DELTA}"
        def PE = "PE_${param.PROBABILITY_EPS}"
        def PT = "PT_${param.PROBABILITY_ETA}"
        def seed = "seed_${param.SYSTEM_RANDOM_SEED}"

        def parameterName = "${SC}_${CR}_${CA}_${TM}_${TN}_${PR}_${PD}_${PE}_${PT}_${seed}"
        trackCountFileName = "track_$parameterName"
        pheGraphFileName = "graph_$parameterName"
    }

}

