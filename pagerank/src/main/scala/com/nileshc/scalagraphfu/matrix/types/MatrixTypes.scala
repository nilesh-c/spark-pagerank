package com.nileshc.scalagraphfu.matrix.types

import no.uib.cipr.matrix.sparse.{FlexCompRowMatrix, SparseVector}
import no.uib.cipr.matrix.{DenseMatrix, DenseVector, Vector}

case class MatrixElement(row: Int, column: Int, value: Double)

case class BlockCoordinates(row: Long, column: Long)

case class BlockMatrixElement(blockCoord: BlockCoordinates, rows: Int, columns: Int) {
  val matrix = new FlexCompRowMatrix(rows, columns)

  def set(row: Int, column: Int, value: Double) = {
    matrix.set(row, column, value)
  }

  def getRowSumsVector(blockVectorElement: BlockRowVectorElement[SparseVector]): BlockRowVectorElement[SparseVector] = {
    var i = 0
    val rows = matrix.numRows()
    while (i < rows) {
      blockVectorElement.set(i, matrix.getRow(i).norm(Vector.Norm.One))
      i += 1
    }
    blockVectorElement
  }

  def elementWiseDivide(blockVectorElement: BlockRowVectorElement[SparseVector]): BlockMatrixElement = {
    val iter = blockVectorElement.vector.iterator()
    while(iter.hasNext) {
      val vectorElement = iter.next()
      // divide row by row sum vector
      matrix.getRow(vectorElement.index()).scale(1.0 / vectorElement.get())
    }
    this
  }

  def *(blockVector: BlockRowVectorElement[DenseVector]): BlockRowVectorElement[DenseVector] = {
    mult(blockVector)
  }

  def mult(blockVector: BlockRowVectorElement[DenseVector]): BlockRowVectorElement[DenseVector] = {
    val newVector = new DenseVector(blockVector.vector.size())
    val denseMatrix = new DenseMatrix(matrix)
    denseMatrix.mult(blockVector.vector, newVector)
    DenseBlockRowVectorElement(blockVector.blockColumn, newVector)
  }

  def transpose(): BlockMatrixElement = {
    matrix.transpose()
    this
  }

  override def toString: String = blockCoord + ",\n" + matrix
}

abstract class BlockRowVectorElement[T <: Vector](val blockColumn: Long, val vector: T) {
  def set(column: Int, value: Double) {
    vector.set(column, value)
  }

  def getNewVector(size: Int): T

  def add(blockVector: BlockRowVectorElement[T]): BlockRowVectorElement[T] = {
    vector.add(blockVector.vector)
    this
  }

  def +(blockVector: BlockRowVectorElement[T]): BlockRowVectorElement[T] = {
    add(blockVector)
  }

  def subtract(blockVector: BlockRowVectorElement[T]): BlockRowVectorElement[T] = {
    vector.add(-1.0, blockVector.vector)
    this
  }

  def -(blockVector: BlockRowVectorElement[T]): BlockRowVectorElement[T] = {
    subtract(blockVector)
  }

  def abs: BlockRowVectorElement[T] = {
    val iter = vector.iterator()
    while(iter.hasNext) {
      val element = iter.next()
      val value = element.get()
      if(value < 0.0 )
        element.set(-value)
    }
    this
  }

  def scale(factor: Double): BlockRowVectorElement[T] = {
    vector.scale(factor)
    this
  }

  def getZeroNodes(): BlockRowVectorElement[T]

  def getNorm1 = vector.norm(Vector.Norm.One)

  def dot(vector: BlockRowVectorElement[_ <: Vector]): Double = {
    this.vector.dot(vector.vector)
  }

  protected def getZeroNodeInternalVector(): T = {
    val size = vector.size()
    var i = 0
//  val newVector = vector match {
//      case v: DenseVector => new DenseVector(v.size())
//      case v: SparseVector => new SparseVector(v.size())
//    }
    val newVector = getNewVector(size)
    while(i < size) {
      if(0.0 == vector.get(i)) newVector.set(i, 1)
      i += 1
    }
    newVector
  }

  override def toString: String = "blockColumn=" + blockColumn + ",\n" + vector
}

case class DenseBlockRowVectorElement(override val blockColumn: Long, override val vector: DenseVector) extends BlockRowVectorElement[DenseVector](blockColumn, vector) {
  def this(blockColumn: Long, columns: Int) = this(blockColumn, new DenseVector(columns))

  override def getZeroNodes(): DenseBlockRowVectorElement = DenseBlockRowVectorElement(blockColumn, super.getZeroNodeInternalVector())

  override def getNewVector(size: Int): DenseVector = new DenseVector(size)
}

case class SparseBlockRowVectorElement(override val blockColumn: Long, override val vector: SparseVector) extends BlockRowVectorElement[SparseVector](blockColumn, vector) {
  def this(blockColumn: Long, columns: Int) = this(blockColumn, new SparseVector(columns))

  override def getZeroNodes(): SparseBlockRowVectorElement = SparseBlockRowVectorElement(blockColumn, super.getZeroNodeInternalVector())

  override  def getNewVector(size: Int): SparseVector = new SparseVector(size)

  def compact(): SparseBlockRowVectorElement = {
    vector.compact()
    this
  }
}

object DenseBlockRowVectorElement {
  def apply(blockColumn: Long, columns: Int) = new DenseBlockRowVectorElement(blockColumn, new DenseVector(columns))
}

object SparseBlockRowVectorElement {
  def apply(blockColumn: Long, columns: Int) = new SparseBlockRowVectorElement(blockColumn, new SparseVector(columns))
}
