package com.nileshc.scalagraphfu.pagerank

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import com.nileshc.scalagraphfu.matrix.types.{BlockRowVectorElement, DenseBlockRowVectorElement, BlockMatrixElement}
import no.uib.cipr.matrix.DenseVector
import no.uib.cipr.matrix.sparse.SparseVector
import org.apache.spark.storage.StorageLevel

/**
 * Created by nilesh on 8/5/14.
 */
object PageRankMatVec {
  /**
   * Performs a single PageRank iteration.
   *
   * @param transitionMatrix Row-normalized blocked transition matrix
   * @param rankVector Current unnormalized PageRank vector
   * @param danglingNodeVector Dangling Node vector
   * @param danglingRankDotProduct Rank vector [dot] Danling node vector
   * @param alpha Damping factor
   * @param blockSize Matrix/Vector block size
   * @param computeL1NormDiff If this is true, the l1norm of the difference between old and new PageRank vectors is returned, else returns 0.0 for l1Diff
   * @return Tuple3(newRankVector, l1Diff, danglingDotProduct)
   */
  def run(transitionMatrix: RDD[(Long, BlockMatrixElement)],
          rankVector: RDD[(Long, BlockRowVectorElement[DenseVector])],
          danglingNodeVector: RDD[(Long, BlockRowVectorElement[SparseVector])],
//          rankVectorSum: Double,
          danglingRankDotProduct: Double,
          alpha: Double,
          blockSize: Int,
          numNodes: Long,
          computeL1NormDiff: Boolean = true)(implicit sc: SparkContext): (RDD[(Long, BlockRowVectorElement[DenseVector])], Double, Double) = {

//    val rankVectorSumBC = sc.broadcast(rankVectorSum)
    // All 1/N teleportationVector - default for PageRank
    val teleportationVector = new DenseVector(Array.fill(blockSize)((alpha * danglingRankDotProduct + 1 - alpha) / numNodes))
    val alphaDanglingFactorVectorBC = sc.broadcast(DenseBlockRowVectorElement(-1, teleportationVector))

    // Setup accumulators
//    val vectorSumAcc = sc.accumulator(0.0)
    val l1DiffAcc = sc.accumulator(0.0)
    val danglingRankDotProductAcc = sc.accumulator(0.0)

    // TODO: The map below can be taken to line 32 : val (matrix, vector) = (matrixVector._1, matrixVector._2.scale(1.0 / rankVectorSumBC.value)) to minimize number of mapreduce tasks.
    // But Spark probably optimizes this with its DAGs. Lesser FLOPs. Maybe benchmark this?
//    val normalizedRankVector = rankVector.map(x => (x._1, x._2.scale(1.0 / rankVectorSumBC.value)))

    // Get the partial vectors and group them by key - to be summed up.
    // TODO: Can also reduceByKey and get the newRankVector instead of newPartialVectors. Benchmark?
    val newPartialVectors = transitionMatrix.join(rankVector).map {
      case (key: Long, matrixVector: (BlockMatrixElement, BlockRowVectorElement[DenseVector])) =>
        val (matrix, vector) = (matrixVector._1, matrixVector._2)
        val newVector = matrix * vector
//        println(matrix + "\n" + vector + "\n" + newVector)
        (matrix.blockCoord.column, newVector)
    }.groupByKey()

    val newRankVector = newPartialVectors.join(rankVector.join(danglingNodeVector)).map{
      case (key: Long, partialVectors_normalizedOldRankVector: (Seq[BlockRowVectorElement[DenseVector]], (BlockRowVectorElement[DenseVector], BlockRowVectorElement[SparseVector]))) =>
        val partialVectors = partialVectors_normalizedOldRankVector._1
        val oldVectorElement = partialVectors_normalizedOldRankVector._2._1
        val danglingNodeVectorElement = partialVectors_normalizedOldRankVector._2._2

        // Sum up partial elements to get the actual rank vector elements.
        val newVectorElement = partialVectors.reduce(_ + _).scale(alpha) + alphaDanglingFactorVectorBC.value
//        vectorSumAcc += newVectorElement.getNorm1
//        println("NEWVECTORELEMENT" + key + "\n" + newVectorElement)
        danglingRankDotProductAcc += newVectorElement dot danglingNodeVectorElement

        if(computeL1NormDiff) l1DiffAcc += (oldVectorElement - newVectorElement).getNorm1

        (key, newVectorElement)
    }

    if(computeL1NormDiff){
      newRankVector.persist(StorageLevel.MEMORY_AND_DISK_SER)
      if(newRankVector.getCheckpointFile.isDefined)
        newRankVector.checkpoint()
      newRankVector.count()
    } // Need to force RDD computation because we need l1DiffAcc to be evaluated.

//    (newRankVector, vectorSumAcc.value, l1DiffAcc.value, danglingRankDotProductAcc.value)
    (newRankVector, l1DiffAcc.value, danglingRankDotProductAcc.value)
  }
}