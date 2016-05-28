import org.apache.spark._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import scala.util.Random

object DSGD {

	def stratPerms (numBlocks:Int) : Iterator[List[Int]] = {
		val x = 0 to numBlocks - 1 toList
		return x.permutations
	}

	// Assign each element of V to a block
	def assignBlocks (row:Int, col:Int, numWorkers:Int) = {
		return (row/numWorkers, col/numWorkers)
	}



	// CONSTANTS
	val NumWorkers = 5
	val Rank = 10
	val maxIter = 100
	val tau = 100
	val beta = -0.6
	val lambda = 0.1 // Regularization parameter

	// load in V text file
	val trainDat = sc.textFile("cme323_final_project/ratings-tiny.txt")
	val entries : RDD[MatrixEntry] = trainDat.map(_.split(",") match { case Array(label,idx,value) => 
		MatrixEntry(label.toInt, idx.toInt, value.toDouble)})
	val V: CoordinateMatrix = new CoordinateMatrix(entries)

	

	val numTrainRow = V.numRows()
	val numTrainCol = V.numCols()

	/*
	// Initialize W and H as random matrices

	val DataW: RDD[Vector] = RandomRDDs.normalVectorRDD(sc, m, Rank, 5, 1)
	val W = new RowMatrix(data, m, Rank) 

	val DataH: RDD[Vector] = RandomRDDs.normalVectorRDD(sc, Rank, n, 5, 1)
	val H = new RowMatrix(data, Rank, n) 

	// Initialize W and H as org.apache.spark.mllib.linalg.Matrix
	var W: Matrix = Matrices.dense(numTrainRow, Rank, Array.fill(numTrainRow*Rank){Random.nextDouble})
	var H: Matrix = Matrices.dense(Rank, numTrainCol, Array.fill(Rank*numTrainCol){Random.nextDouble})
	
	*/

	// Initialize W and H as random 2D arrays (matrices)
	val W: Array[Array[Double]] = Array.fill(numTrainRow, Rank) { Random.nextDouble }
	val H: Array[Array[Double]] = Array.fill(Rank, numTrainCol) { Random.nextDouble }
	val HTransposed = H.transpose

	// Broadcast V (V), W, and H

	val bcV = sc.broadcast(V)
	val bcW = sc.broadcast(W)
	val bcH = sc.broadcast(HTransposed)


	// Loop while not converge
	// Loop permutation to pick Strata
	// Run SGD on blocks in parallel

	// SGD updates for a single block

	var iter = 0

	val strata = stratPerms(numWorkers)

	while (iter < maxIter) {
		val stepSize = scala.math.pow((tau + iter),beta)
		while (strata.hasNext) {
			var rowID = 0
			for (colID <- strata.next) { //change this to parallel for
				blockVEntries = V.entries.filter(entry => 
					((assignBlocks(entry.i, entry.j, numWorkers)._1 == rowID) && 
					(assignBlocks(entry.it, entry.j, numWorkers)._2 == colID))) //Change this to broadcast variable
				for (entry <- blockVEntries) {
					val WRow = W(entry.i)
					//val HCol = H.map{_(entry.j)}
					val Hcol = HTransposed(entry.j)
					val VMinusWH = 2*(entry.value - (WRow,HCol).zipped.map(_*_).sum) 
					val gradW = (Hcol.map(_*(-VMinusWH)), WRow.map(_*2*lambda)).zipped.map(_+_)
					W(entry.i) = (WRow,gradW).zipped.map(_-stepSize*_)

					val gradH = (WRow.map(_*(-VMinusWH)), HCol.map(_*2*lambda)).zipped.map(_+_)
					HTransposed(entry.j) = (Hcol,gradH).zipped.map(_-stepSize*_)

				}
				rowID = rowID + 1
			}
		}

		iter = iter + 1
	}





}