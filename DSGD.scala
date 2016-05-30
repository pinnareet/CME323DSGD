import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.rdd.RDD
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

	def assignBlockIndex (index: Int, numData: Int, numWorkers: Int) = {
		Math.floor(index/Math.ceil(numData/numWorkers)).toInt
	}


	// CONSTANTS
	val numWorkers = 2
	val Rank = 4
	val maxIter = 100
	val tau = 100
	val beta = -0.6
	val lambda = 0.1 // Regularization parameter

	// load in V text file
	//val trainDat = sc.textFile("cme323_final_project/ratings-tiny.txt")
	val trainDat = sc.textFile("cme323_final_project/test.txt")
	val entries : RDD[MatrixEntry] = trainDat.map(_.split(",") match { case Array(label,idx,value) => 
		MatrixEntry(label.toInt, idx.toInt, value.toDouble)})
	val V: CoordinateMatrix = new CoordinateMatrix(entries)

	

	val numTrainRow = V.numRows().toInt
	val numTrainCol = V.numCols().toInt

	// Initialize W and H as random 2D arrays (matrices)

	//var W = sc.parallelize(0 to numTrainRow-1).map(x => (x,Array.fill(numTrainRow) { Random.nextDouble }))
	var W = sc.parallelize(0 to numTrainRow-1).map(x => (x,Array.fill(numTrainRow) { 0.1 * (x%13)}))
	// Have NumWorkers work on this parallelize (range, NumWorkers)
	
	//var HT = sc.parallelize(0 to numTrainCol-1).map(x => (x,Array.fill(numTrainCol) { Random.nextDouble }))
	var H = sc.parallelize(0 to numTrainCol-1).map(x => (x,Array.fill(numTrainCol) { 0.1 * (x%11)}))

	//val VRDD = V.entries.map(entry => (assignBlockIndex(entry.i.toInt, numTrainRow, numWorkers), entry))
	var WBlocked = W.map(tuple => (assignBlockIndex(tuple._1, numTrainRow, numWorkers), tuple))
	var HTBlocked = HT.map(tuple => (assignBlockIndex(tuple._1, numTrainRow, numWorkers), tuple))


	var iter = 0

	var strata = stratPerms(numWorkers)

	def stratTuples (strata: List[Int]) = {
		val len = strata.length
		val i = List.range(0,len)
		i.map(a => (a, strata(a)))
	}

	//val nnGroup = newGroup.flatMap(x => )	

	//val mew = res10.map(entry => if (entry._2 == 3) (entry._1, 8) else entry) 
	
	// Returns an RDD with updated value of W and H
	// RDD[(Int, (Iterable[org.apache.spark.mllib.linalg.distributed.MatrixEntry], Iterable[(Int, Array[Double])], Iterable[(Int, Array[Double])]))]

	// vwh is a block of V and corresponding W and H. Returns updated RDD of W and H 
	def SGD (vwh : (Int, (Iterable[MatrixEntry], Iterable[(Int, Array[Double])], Iterable[(Int, Array[Double])]))) = {
		val VTest : Iterator[MatrixEntry] = vwh._2._1.iterator
		var WTest : Iterator[(Int, Array[Double])] = vwh._2._2.iterator
		var HTest : Iterator[(Int, Array[Double])]= vwh._2._3.iterator
		//update W and H 
		//W = ...
	}

	val numStrata = strata.length

	while (iter < maxIter) { //Or until convergence
		val stepSize = scala.math.pow((tau + iter),beta)
		val colPerms = strata(iter % numStrata) 

		//Build a set of strata
		val VRDD = V.entries.filter(entry => stratTuples(colPerms).contains((assignBlockIndex(entry.i.toInt, numTrainRow, numWorkers)
			, assignBlockIndex(entry.j.toInt, numTrainCol, numWorkers))))

		val keyedVRDD = VRDD.map(entry => (assignBlockIndex(entry.i.toInt, numTrainRow, numWorkers),entry))

		val HTPermBlocked = HTBlocked.map(tuple => (colPerms(tuple._1), tuple._2))

		val VWH = keyedVRDD.cogroup(WBlocked, HTPermBlocked)

		VWH.partitionBy(new HashPartitioner(numWorkers))

		val updatedWH = group.mapPartitions(a => a.map(b => SGD(b))) //Each partition has 1 element RDD

		//val updatedVWH = VWH.mapPartitions(a => SGD(a)) //update W and H 

		//val mew = res10.map(entry => if (entry._2 == 3) (entry._1, 8) else entry) 


		/*
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
		*/

		iter = iter + 1
	}


