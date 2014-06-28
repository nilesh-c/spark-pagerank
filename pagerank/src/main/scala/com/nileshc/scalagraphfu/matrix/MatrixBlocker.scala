package com.nileshc.scalagraphfu.matrix

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import com.nileshc.scalagraphfu.matrix.types.{MatrixElement, BlockCoordinates, BlockMatrixElement}
import org.apache.spark.SparkContext

/**
* Created by nilesh on 7/5/14.
*/
object MatrixBlocker {
  /**
   * Takes the edge triples of the adjacency matrix and build a 2D-blocked matrix.
   *
   * @param sparseMatrixFileLines Edge triples of adjacency matrix
   * @param blockSize Matrix/Vector block size
   * @return Blocked matrix
   */
  def run(sparseMatrixFileLines: RDD[String], blockSize: Int)(implicit sc: SparkContext) : RDD[(Long, BlockMatrixElement)] = {
    val blockSizeBC = sc.broadcast(blockSize)

    // Get edges from the edge file and put them into blocks.
    val edgeTriples = sparseMatrixFileLines.map(x => {
      val blockSize = blockSizeBC.value
      val line = x.split(""",""")
      val row = line(0).toLong
      val column = line(1).toLong
      val value = line(2).toDouble
      (BlockCoordinates(row / blockSize, column / blockSize), MatrixElement((row % blockSize).toInt, (column % blockSize).toInt, value))
    })

    // Create blocked matrix elements and key them by row.
    val blockedMatrix = edgeTriples.groupByKey().map{case (blockCoords: BlockCoordinates, matrixElements: Seq[MatrixElement]) =>
      val blockMatrixElement = BlockMatrixElement(blockCoords, blockSize, blockSize)
      matrixElements.foreach(e => blockMatrixElement.set(e.row, e.column, e.value))
      (blockCoords.row, blockMatrixElement)
    }

    blockedMatrix
  }
}

//object MatrixBlocker {
//  def apply(blockSize: Int) = new MatrixBlocker(blockSize)
//}