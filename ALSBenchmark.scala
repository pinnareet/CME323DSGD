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
val trainDat =sc.textFile("/FileStore/tables/l2sz6xo11464326685589/TrainingRatings.txt")
val testDat = sc.textFile("/FileStore/tables/szv990pk1464328768773/TestingRatings.txt")

/**get train ratings RDD**/
val ratings = trainDat.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

/** run algorithm from ALS implementation and time it for different ranks of the factors **/

val ranks = Vector(5,10,15,20,30,50)
val numIterations = 10
for(rank <- ranks)
{
  val model = time{ALS.train(ratings, rank, numIterations, 0.01)}
  
  val testRat = testDat.map(_.split(',') match { case Array(user, item, rate) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})
  val usersMovies = testRat.map{ case Rating(user, movie, rate) =>
  (user, movie)}
  val predictions = model.predict(usersMovies).map { case Rating(user, movie, rate) =>
    ((user, movie), rate)}
  val ratesAndPreds = testRat.map{ case Rating(user, movie, rate) =>
  ((user, movie), rate)}.join(predictions)
  val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)
}
