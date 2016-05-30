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
		val VIter : Iterator[MatrixEntry] = vwh._2._1.iterator
		var WIter : Iterator[(Int, Array[Double])] = vwh._2._2.iterator 
		var HIter : Iterator[(Int, Array[Double])]= vwh._2._3.iterator

		for (entry <- VIter) {
			val rowID : Int = entry.i.toInt
			val colID : Int = entry.j.toInt

			val WRow = WIter.filter(row => row._1 == rowID).toList(0)._2
			val HCol = HIter.filter(col => col._1 == colID).toList(0)._2

			val VMinusWH = 2*(entry.value - (WRow,HCol).zipped.map(_*_).sum) 

			val gradW = (HCol.map(_*(-VMinusWH)), WRow.map(_*2*lambda)).zipped.map(_+_)
			WIter = WIter.map(x => if (x._1 == rowID) (x._1, (WRow,gradW).zipped.map(_-stepSize*_)) else x)
			
			//W(entry.i.toInt) = (WRow,gradW).zipped.map(_-stepSize*_)

			val gradH = (WRow.map(_*(-VMinusWH)), HCol.map(_*2*lambda)).zipped.map(_+_)
			HIter = HIter.map(x => if (x._1 == colID) (x._1, (HCol,gradH).zipped.map(_-stepSize*_)) else x)
			//HTransposed(entry.j.toInt) = (HCol,gradH).zipped.map(_-stepSize*_)
		}
		
		(WIter, HIter)
	}

	val numStrata = strata.length

	var WH : RDD[(Iterator[(Int, Array[Double])], Iterator[(Int, Array[Double])])] = sc.emptyRDD[(Iterator[(Int, Array[Double])], Iterator[(Int, Array[Double])])]
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

		WH = VWH.mapPartitions(a => a.map(b => SGD(b))) //Each partition has 1 element RDD

		//updatedWH updates W and H and become an RDD of (Iterator[(Int, Array[Double])], Iterator[(Int, Array[Double])])
		iter = iter + 1
	}



// Extract W and H 
