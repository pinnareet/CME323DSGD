// Preprocesses the text ratings data to have consecutive user and movie IDs. 

import org.apache.spark._

def time[R](block: => R): R = {  
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0)/1e9 + "s")
    result
}


val triples = sc.textFile("ratings-tiny.txt")

/*
var user = Array[String]()
var movie = Array[String]()
var rating = Array[String]()
val parsed = triples.map(_.split(","))
val rowIndex = parsed.map(a => user :+ a(0)).map(a=>a(0).toInt)
val columnIndex = parsed.map(a => movie :+ a(1)).map(a=>a(0).toInt)
val values = parsed.map(a => rating :+ a(2)).map(a=>a(0).toInt)
*/


val entries : RDD[MatrixEntry] = triples.map(_.split(",") match { case Array(label,idx,value) => MatrixEntry(label.toInt, idx.toInt, value.toDouble)})
val mat: CoordinateMatrix = new CoordinateMatrix(entries)

