/*
 * @author Sagar Shukla
 */

import java.util.*;
import java.io.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

// KMeans Algorithm implementation. 
public class KMeansAlgo {
	public static String IN;
	public static String OUT;
	public static String INPUT;
	public static String OUTPUT;
	public static String AGAIN_INPUT;
	public static String CENTROID_FILE_NAME = "/centroids.txt";
	public static String POINTS_FILE_NAME = "/points.txt";
	public static String OUTPUT_FILE_NAME = "/part-00000";
	public static List<Double> centroids = new ArrayList<Double>();
	
	
	// KMeansAlgo Mapper class.
	public static class KMeansMapper extends MapReduceBase implements Mapper<LongWritable, Text, DoubleWritable, DoubleWritable>{
		
		/* Here we are overriding the configure function to get all the centroids
		 * from the file.
		 */
		@Override
		public void configure(JobConf job) {
			try {
				Path[] cacheFiles = DistributedCache.getLocalCacheFiles(job);
				if(cacheFiles != null && cacheFiles.length > 0) {
					String theLine;
					centroids.clear();
					
					BufferedReader br = new BufferedReader(new FileReader(cacheFiles[0].toString()));
					try {
						while((theLine = br.readLine()) != null) {
							String[] tempCentrod = theLine.split("\t| ");
							centroids.add(Double.parseDouble(tempCentrod[0]));
						}
					}
					
					finally {
						br.close();
					}
				}
			}
			
			catch(IOException e) {
				System.err.println("Exception reading DistribtuedCache: " + e);
			}
		}

		/* Map function is used to find the nearest centre of each point
		 * given in the file.
		 */
		@Override
		public void map(LongWritable key, Text value, OutputCollector<DoubleWritable, DoubleWritable> output,
				Reporter report) throws IOException {
			
			String line = value.toString();
			double point = Double.parseDouble(line);
			double min1;
			double min2 = Double.MAX_VALUE;
			double min_centre = centroids.get(0);
			
			for(double centre : centroids) {
				min1 = centre - point;
				
				if(Math.abs(min1) < Math.abs(min2)) {
					min_centre = centre;
					min2 = min1;
				}
			}
			
			output.collect(new DoubleWritable(min_centre), new DoubleWritable(point));
		}
	}
	
	
	// KMeansAlgo Reducer class.
	public static class KMeansReducer extends MapReduceBase implements Reducer<DoubleWritable, DoubleWritable, DoubleWritable, Text>{
		
		/* reduce function is simply calculating the new centroids.*/
		@Override
		public void reduce(DoubleWritable key, Iterator<DoubleWritable> values,
				OutputCollector<DoubleWritable, Text> output, Reporter report) throws IOException {
			double newCentre;
			double sumOfPoints = 0;
			int noOfElements = 0;
			String points = "";
			
			while(values.hasNext()) {
				double temp = values.next().get();
				points = points + " " + Double.toString(temp);
				sumOfPoints += temp;
				noOfElements++;
			}
			
			newCentre = sumOfPoints/noOfElements;
			
			output.collect(new DoubleWritable(newCentre), new Text(points));	
		}
	}
	
	
	// Driver Function.
	public static void runKMeans(String[] args) throws Exception{
		IN = args[0];
		OUT = args[1];
		INPUT = IN;
		OUTPUT = OUT + System.nanoTime();;
		AGAIN_INPUT = OUTPUT ;
		
		// Iterating till the termination condition
		int iteration_no = 0;
		boolean isDone = false;
		
		while(isDone == false) {
			JobConf conf = new JobConf(KMeansAlgo.class);
			
			if(iteration_no == 0) {
				Path pathForCentroids = new Path(INPUT + CENTROID_FILE_NAME);
				DistributedCache.addCacheFile(pathForCentroids.toUri(), conf);
			}
			
			else {
				Path pathForCentroids = new Path(AGAIN_INPUT + OUTPUT_FILE_NAME);
				DistributedCache.addCacheFile(pathForCentroids.toUri(), conf);
			}
			
			// Configure the new Job.
			conf.setJobName("KMeansAlgo");
			conf.setMapOutputKeyClass(DoubleWritable.class);
			conf.setMapOutputValueClass(DoubleWritable.class);
			conf.setOutputKeyClass(DoubleWritable.class);
			conf.setOutputValueClass(Text.class);
			conf.setMapperClass(KMeansMapper.class);
			conf.setReducerClass(KMeansReducer.class);
			conf.setInputFormat(TextInputFormat.class);
			conf.setOutputFormat(TextOutputFormat.class);
			FileInputFormat.setInputPaths(conf, new Path(INPUT + POINTS_FILE_NAME));
			FileOutputFormat.setOutputPath(conf, new Path(OUTPUT));
			JobClient.runJob(conf);
			
			/* From line no 144 to 156, we are reading the new centroids
			 * from the new output file and storing them in the new data-structure
			 * i.e. List.
			 */
			FileSystem fs = FileSystem.get(new Configuration());
			BufferedReader br = new BufferedReader(
					new InputStreamReader(fs.open(new Path(OUTPUT + OUTPUT_FILE_NAME))));
			
			List<Double> centroids_next = new ArrayList<Double>();
			String line = br.readLine();
			
			while(line != null){
				String[] temp = line.split("\t| ");
				centroids_next.add(Double.parseDouble(temp[0]));
				line = br.readLine();
			}
			br.close();
			
			
			/* From line no 163 to 184, we are reading the centroids 
			 * from the previous output file and stroing them in the new
			 * data-structure i.e. List.
			 */
			String previous;
			if(iteration_no == 0) {
				previous = INPUT + CENTROID_FILE_NAME;
			}
			
			else {
				previous = AGAIN_INPUT + OUTPUT_FILE_NAME;
			}
			
			FileSystem fs1 = FileSystem.get(new Configuration());
			
			BufferedReader br1 = new BufferedReader(
					new InputStreamReader(fs.open(new Path(previous))));	
			List<Double> centeroids_prev = new ArrayList<Double>();
			String line1 = br1.readLine();
			
			while(line1 != null) {
				String[] temp = line1.split("\t| ");
				centeroids_prev.add(Double.parseDouble(temp[0]));
				line1 = br1.readLine();
			}
			br1.close();
			
			// Sorting the old centroid and new centroid and checking for convergence
			// condition
			
			Collections.sort(centroids_next);
			Collections.sort(centeroids_prev);
			
			Iterator<Double> it = centeroids_prev.iterator();
			
			for(double i : centroids_next) {
				double temp = it.next();
				if(Math.abs(i - temp) <= 0.1) {
					isDone = true;
				}
				
				else {
					isDone = false;
					break;
				}
			}
			
			iteration_no++;
			AGAIN_INPUT = OUTPUT;
			OUTPUT = OUT + System.nanoTime();
		}
	}
	
	// Main Function
	public static void main(String[] args) throws Exception{
		runKMeans(args);
	}
}
