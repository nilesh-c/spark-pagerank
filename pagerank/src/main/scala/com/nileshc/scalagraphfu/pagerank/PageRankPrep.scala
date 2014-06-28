package com.nileshc.scalagraphfu.pagerank

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import com.nileshc.scalagraphfu.matrix.types.{DenseBlockRowVectorElement, SparseBlockRowVectorElement, BlockRowVectorElement, BlockMatrixElement}
import no.uib.cipr.matrix.DenseVector
import no.uib.cipr.matrix.sparse.SparseVector
import org.apache.spark.SparkContext

/**
 * Created by nilesh on 8/5/14.
 */
object PageRankPrep {
  /**
   * Normalizes the blocked matrix, builds the blocked dangling node vector and initial page rank vector.
   *
   * @param blockedMatrix Blocked matrix
   * @param blockSize Matrix/Vector block size
   * @return (normalizedMatrix, initialRankVector, danglingNodeVector, danglingRankDotProduct)
   */
  def run(blockedMatrix: RDD[(Long, BlockMatrixElement)],
          blockSize: Int,
          numNodes: Long)(implicit sc: SparkContext): (RDD[(Long, BlockMatrixElement)],
    RDD[(Long, BlockRowVectorElement[DenseVector])],
    RDD[(Long, BlockRowVectorElement[SparseVector])],
    Double) = {

    val blockSizeBC = sc.broadcast(blockSize)
    val danglingRankDotProductAcc = sc.accumulator(0.0)

    // create partial row sum vectors and reduce/add them by key.
    val rowSumVectors = blockedMatrix.map {
      case (key: Long, blockMatrixElement: BlockMatrixElement) =>
        val rowSums = blockMatrixElement.getRowSumsVector(SparseBlockRowVectorElement(key, blockSizeBC.value).compact())
        (key, rowSums)
    }.reduceByKey(_ + _)

    // rank vector will be a vector of all ones, with sum = numNodes - will be normalized in PageRankMatVec
    val initialRankMTJVectorBC = sc.broadcast(new DenseVector(Array.fill(blockSize)(1.0 / numNodes)))

    // compute initial vector
    val initialRankVector = rowSumVectors.map {
      case (key: Long, rowSumVector: BlockRowVectorElement[_]) =>
        (key, DenseBlockRowVectorElement(key, initialRankMTJVectorBC.value).asInstanceOf[BlockRowVectorElement[DenseVector]])
    }

    // compute dangling node vector
    val danglingNodeVector = rowSumVectors.map {
      case (key: Long, rowSumVector: BlockRowVectorElement[SparseVector]) =>
        (key, rowSumVector.getZeroNodes())
    }

    // compute dot product of initial rank vector and dangling vector
    initialRankVector.join(danglingNodeVector).map {
      case (key: Long, vectors: (BlockRowVectorElement[DenseVector], BlockRowVectorElement[SparseVector])) =>
        danglingRankDotProductAcc += vectors._1.dot(vectors._2)
    }

    val normalizedMatrix = blockedMatrix.join(rowSumVectors).map {
      case (key: Long, matrixVector: (BlockMatrixElement, BlockRowVectorElement[SparseVector])) =>
        val matrix = matrixVector._1
        val vector = matrixVector._2
        // We are building transposed block matrix elements.
        // Faster to multiply with the block vector element directly rather than multiplying with the transpose computed in real-time.
        // TODO: Is this really faster? Or should we use transMult in MatVec? Benchmark.
        (key, matrix.elementWiseDivide(vector).transpose())
    }

    (normalizedMatrix, initialRankVector, danglingNodeVector, danglingRankDotProductAcc.value)
  }
}