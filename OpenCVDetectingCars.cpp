/*
STEP-1:

Download the image database from here : http://cogcomp.cs.illinois.edu/Data/Car/

save it on desktop

STEP-2:


create two folders Desktop viz. pos and neg

1.Copy all the positive images in a folder named pos.
Copy all the negative images in a folder named neg from TrainImages folder.

2.Create a folder named data to store cascade file generated later on.

STEP-3:

Open a terminal,navigate to the requires folder and type

1.find pos -iname "*.pgm" -exec echo \{\} 1 0 0 100 40 \; > cars.info

which will create file cars.info with details

2. find neg -iname "*.pgm" > bg.txt

which will create file bg.txt with details

3. create vector file from positive images (pos folder i.e. cars.info).
opencv_createsamples -info cars.info -num 550 -w 48 -h 24 -vec cars.vec
 
 (width and height parameters change with change of database,-num is the number of images in pos folder)

STEP-4:  Train the cascade: 

opencv_traincascade -data data -vec cars.vec -bg bg.txt -numStages 10 -nsplits 2 -minhitrate 0.999 -maxfalsealarm 0.5 -numPos 500 -numNeg 500 -w 48 -h 24


-featureType LBP

you can use LBP/Haar

Note :

numPos < number of samples in vec


STEP-4:  Test the cascade

use following simple program:

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

String car_cascade_name = "/home/user/Desktop/data/cascade.xml";

CascadeClassifier car_cascade;

string window_name = "Capture ";

 int main( int argc, const char** argv )
 {
    Mat frame;

    if( !car_cascade.load( car_cascade_name ) ){
        printf("--(!)Error loading\n"); return -1;
    };
  
    frame = imread(argv[1]);

    //-- 3. Apply the classifier to the frame
    if( !frame.empty() )
        detectAndDisplay( frame );    
    else
        printf(" --(!) No captured frame -- Break!");
    waitKey(0);
}
   
 
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> cars;
    Mat frame_gray;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect cars
    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    //detectMultiScale: Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.

      for( vector<Rect>::const_iterator main = cars.begin(); main != cars.end(); main++)
        {
        int x0 = cvRound(main->x);                             
        int y0 = cvRound(main->y);
        int x1 = cvRound((main->x + main->width-1));
        int y1 = cvRound((main->y + main->height-1));
       
        rectangle( frame, cvPoint(x0,y0), cvPoint(x1,y1), CV_RGB(0, 255,0), 3, 8, 0);          
    }
    imshow( window_name, frame );
 }
 /*
--------------------------------------------------------------------------------------------------------------------
you can also refer:

http://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html

https://abhishek4273.com/2014/03/16/traincascade-and-car-detection-using-opencv/

https://www.youtube.com/watch?v=WEzm7L5zoZE

http://scholarpublishing.org/index.php/AIVP/article/view/1152/626

https://www.behance.net/gallery/Vehicle-Detection-Tracking-and-Counting/4057777

http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html */
