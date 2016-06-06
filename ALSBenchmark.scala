import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating


def time[R](block: => R): R = {  
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/1e9 + "s")
    result
}


/**load training and testing data**/
/**val trainDat = sc.textFile("/FileStore/tables/8j3z7jgi1465112138947/RatingsShuf.txt")*/
val trainDat = sc.textFile("/FileStore/tables/7dn7w4ew1465114785307/RatingsShuf.txt")

/**val testDat = sc.textFile("/FileStore/tables/szv990pk1464328768773/TestingRatings.txt")**/

/**get train ratings RDD**/
val ratings = trainDat.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

/** run algorithm from ALS implementation and time it for different ranks of the factors **/
/**loop over various factor sizes*/
val ranks = Vector(10,20,50,100,200,400)
val numIterations = 20
for(rank <- ranks)
{
  time{val model = ALS.train(ratings, rank, numIterations, 0.1)
  
  val usersMovies = ratings.map{ case Rating(user, movie, rate) =>
  (user, movie)}
  val predictions = model.predict(usersMovies).map { case Rating(user, movie, rate) =>
    ((user, movie), rate)}
  val ratesAndPreds = ratings.map{ case Rating(user, movie, rate) =>
  ((user, movie), rate)}.join(predictions)
  val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)}
}
