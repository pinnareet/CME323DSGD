import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import scala.util.Random


	def stratPerms (numBlocks:Int) = {
		val x = 0 to numBlocks - 1 toList
		val p = x.permutations.toList
		p
	}

	// Assign each element of V to a block
	def assignBlocks (row:Int, col:Int, numWorkers:Int, numTrainRow: Int, numTrainCol: Int) = {
		val blockRowSize = Math.ceil(numTrainRow/numWorkers)
		val blockColSize = Math.ceil(numTrainCol/numWorkers)
		(Math.floor(row/blockRowSize), Math.floor(col/blockColSize))
	}


	// CONSTANTS
	val numWorkers = 3
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

	

	val numTrainRow = V.numRows().toInt
	val numTrainCol = V.numCols().toInt

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
	var W: Array[Array[Double]] = Array.fill(numTrainRow, Rank) { Random.nextDouble }
	//var W = sc.parallelize(1 to numTrainRow).map(x => (x,Array.fill(numTrainRow) { Random.nextDouble }))
	W(747)(0)
	val oldW = W
	var H: Array[Array[Double]] = Array.fill(Rank, numTrainCol) { Random.nextDouble }
	//var HTransposed = sc.parallelize(1 to numTrainCol).map(x => (x,Array.fill(numTrainCol) { Random.nextDouble }))
	var HTransposed = H.transpose

	// Broadcast V (V), W, and H
	/*
	val bcV = sc.broadcast(V)
	val bcW = sc.broadcast(W)
	val bcH = sc.broadcast(HTransposed)
	*/

	// Loop while not converge
	// Loop permutation to pick Strata
	// Run SGD on blocks in parallel

	// SGD updates for a single block

	var iter = 0

	var strata = stratPerms(numWorkers)
	val numStrata = strata.length

	while (iter < maxIter) {
		val stepSize = scala.math.pow((tau + iter),beta)
		val colPerms = strata(iter % numStrata) 


		for (i <- 0 to numWorkers-1) {
			//val perm = strata.next
			val rowID = i
			val colID = colPerms(i)
			//val colID = strata.next(3)	
				val blockVEntries = V.entries.filter(entry => 
					((assignBlocks(entry.i.toInt, entry.j.toInt, numWorkers, numTrainRow, numTrainCol)._1 == rowID) && 
					(assignBlocks(entry.i.toInt, entry.j.toInt, numWorkers, numTrainRow, numTrainCol)._2 == colID))) //Change this to broadcast variable
				for (entry <- blockVEntries.collect()) { //Change to parallel for
				//val entry = blockVEntries.first
					val WRow = W(entry.i.toInt)
					val HCol = HTransposed(entry.j.toInt)

					val VMinusWH = 2*(entry.value - (WRow,HCol).zipped.map(_*_).sum) 

					val gradW = (HCol.map(_*(-VMinusWH)), WRow.map(_*2*lambda)).zipped.map(_+_)
					W(entry.i.toInt) = (WRow,gradW).zipped.map(_-stepSize*_)

					val gradH = (WRow.map(_*(-VMinusWH)), HCol.map(_*2*lambda)).zipped.map(_+_)
					HTransposed(entry.j.toInt) = (HCol,gradH).zipped.map(_-stepSize*_)

				}
				//rowID = rowID + 1
			
		}
		

		iter = iter + 1
	}

	W(747)(0)


