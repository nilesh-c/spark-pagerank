package com.nileshc.scalagraphfu.pagerank

import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import com.nileshc.scalagraphfu.matrix.types.{BlockRowVectorElement, MatrixElement, BlockCoordinates}
import com.nileshc.scalagraphfu.matrix.MatrixBlocker
import org.apache.spark.rdd.RDD
import no.uib.cipr.matrix.DenseVector
import scala.collection.mutable.ListBuffer

//import com.nileshc.scalagraphfu.matrix.MatrixBlocker

/**
 * Created by nilesh on 2/5/14.
 */
object Main {
  def main(args: Array[String]) {
//    val blockSize = 53
//    val numNodes = 106
//    if(args.length < 1) {
//      println("Usage: mvn scala:run -Dlauncher=Main matrix-edge-file")
//      sys.exit(-1)
//    }
//    val sc = getSparkContext()
//    val blockedMatrix = MatrixBlocker(blockSize).run(sc.textFile(args(0)))
//    val (normalizedMatrix, danglingNodeVector, initialRankVector) = PageRankPrep(blockSize, numNodes).run(blockedMatrix)
    silenceSpark()
    implicit val sc = getSparkContext()
    sc.setCheckpointDir("/home/nilesh/checkpoint")
    val blockSize = 2
    val numNodes = 4
    val alpha = 0.85

    val text = sc.textFile("/home/nilesh/matrix.csv").filter(!_.trim.startsWith("//"))
//    val blockedMatrix = MatrixBlocker.run(text, blockSize)
//    val (normalizedMatrix, initialRankVector, danglingNodeVector, drDotProduct) = PageRankPrep.run(blockedMatrix, blockSize, numNodes)
//    println("NORMALIZED MATRIX:\n" + normalizedMatrix.collect().mkString("\n"))
//    println("danglingNodeVector:\n" + danglingNodeVector.collect().mkString("\n"))
//    println("initialRankVector:\n" + initialRankVector.collect().mkString("\n"))
//    println("danglingRankDotProduct:\n" + drDotProduct)

////    var currentRankVectorSum: Double = 1.0
//    var currentDanglingRankDotProduct: Double = drDotProduct
//    var currentRankVector: RDD[(Long, BlockRowVectorElement[DenseVector])] = initialRankVector

//    val tuple = PageRankMatVec.run(normalizedMatrix, rankVector = currentRankVector, danglingNodeVector, danglingRankDotProduct = currentDanglingRankDotProduct, alpha, blockSize, numNodes, false)
//    currentRankVector = tuple._1
////    currentRankVectorSum = tuple._2
//    val l1Norm = tuple._2
//    currentDanglingRankDotProduct = tuple._3

    println(pageRank(text, alpha, blockSize, numNodes, 80, 0.0001, 10).collect().mkString("\n"))
//
//    println("currentRankVector" + currentRankVector.collect().mkString("\n"))
////    println("currentRankVectorSum" + currentRankVectorSum)
//    println("l1Norm" + l1Norm)
//    println("currentDanglingRankDotProduct" + currentDanglingRankDotProduct)
  }

  def pageRank(sparseMatrixFileLines: RDD[String], alpha: Double, blockSize: Int, numNodes: Int, maxIter: Int = 100, epsilon: Double = 1e-6, iterationCheckInterval: Int = 3)(implicit sc: SparkContext): RDD[(Long, Double)] = {
    val blockedMatrix = MatrixBlocker.run(sparseMatrixFileLines, blockSize)
    val (normalizedMatrix, initialRankVector, danglingNodeVector, drDotProduct) = PageRankPrep.run(blockedMatrix, blockSize, numNodes)
    normalizedMatrix.cache()
//    println(normalizedMatrix.collect().mkString("\n"))

    var hasConverged = false
    var iter = 0

//    var currentRankVectorSum: Double = 1.0
    var currentDanglingRankDotProduct: Double = drDotProduct
    var currentRankVector: RDD[(Long, BlockRowVectorElement[DenseVector])] = initialRankVector

    while(!hasConverged) {
      val checkForConvergence = (iter + 1) % iterationCheckInterval == 0 // we need to check for convergence at particular intervals.
      val tuple = PageRankMatVec.run(normalizedMatrix,
        rankVector = currentRankVector,
        danglingNodeVector,
//        rankVectorSum = currentRankVectorSum,
        danglingRankDotProduct = currentDanglingRankDotProduct,
        alpha,
        blockSize,
        numNodes,
        computeL1NormDiff = checkForConvergence) // compute l1Diff if we need to check for convergence.
      currentRankVector = tuple._1
//      currentRankVectorSum = tuple._2
      val residual = tuple._2
      currentDanglingRankDotProduct = tuple._3

      iter += 1
      if(iter >= maxIter)
        hasConverged = true

      println("Residual=" + residual)
      println("Iterations=" + iter)
      if(checkForConvergence) println("[" + unfoldBlockedVector(currentRankVector).collect().map(_._2).mkString(" ") + "]")
    }

    // Convert RDD of (Long,BlockRowVectorElement) to RDD of (Long,Double).
    unfoldBlockedVector(currentRankVector)
  }

  def unfoldBlockedVector(blockedVector: RDD[(Long, BlockRowVectorElement[DenseVector])]): RDD[(Long, Double)] = {
    blockedVector.flatMap {
      case (key: Long, vectorElement: BlockRowVectorElement[DenseVector]) =>
        var list = new ListBuffer[(Long, Double)]()
        val iter = vectorElement.vector.iterator()
        while (iter.hasNext) {
          val element = iter.next()
          val tuple = (key * element.index().toLong, element.get())
          list += tuple
        }
        list.result()
    }
  }

  /**
   * Set all loggers to the given log level.  Returns a map of the value of every logger
   * @param level
   * @param loggers
   * @return
   */
  def setLogLevels(level: org.apache.log4j.Level, loggers: TraversableOnce[String]) =
  {
    loggers.map
      {
        loggerName =>
          val logger = org.apache.log4j.Logger.getLogger(loggerName)
          val prevLevel = logger.getLevel()
          logger.setLevel(level)
          loggerName -> prevLevel
      }.toMap
  }

  /**
   * Turn off most of spark logging.  Returns a map of the previous values so you can turn logging back to its
   * former values
   */
  def silenceSpark() =
  {
    setLogLevels(org.apache.log4j.Level.WARN, Seq("org.apache", "spark", "org.eclipse.jetty", "akka"))
  }


  def getSparkContext() : SparkContext = {
    val conf = new SparkConf().setMaster("local[8]").setAppName("rank-aggregation")
    conf.setSparkHome("/home/nilesh/utils/spark-0.9.1-bin-hadoop2")
    conf.setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    //conf.set("spark.closure.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    //conf.set("spark.kryo.registrator", "com.nileshc.scalagraphfu.serialize.KryoExtractionRegistrator")
    conf.set("spark.kryoserializer.buffer.mb", "50")
    new SparkContext(conf)
  }
}
