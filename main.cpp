#include <highgui.h>
#include <iostream>
#include <cv.h>
#include <math.h>
#include <float.h>

using namespace std;
using namespace cv;

///Global Var
Mat src_gray;
Mat hasil;
int thresh = 65;
int max_thresh = 255;
RNG rng(12345);
Mat imgOri;
Mat edge ;
vector<vector<double> > dataset;

///file finder

string pathDataTrainIkan = "gambar/train/ikan/test";
string pathDataTrainNonIkan = "gambar/train/nonIkan/test";
/// Function header
void thresh_callback(int, void* );

string number_to_string(int x){
if(!x) return "0";
    string s,s2;
    while(x){
        s.push_back(x%10 + '0');
        x/=10;
    }
reverse(s.begin(),s.end());
return s;
}

///Fitur

double kebulatan;
double kepadatan;
double compactness;
double convexity;

void thresh_callback(int, void* );

void ekstrakFitur(vector<Point> contours){

    vector<Point> hull( contours.size());

    /// OPERASI FITUR YANG DI EXTRAKSI ////////
    convexHull( Mat(contours), hull, false ); // untuk mendapatkan vector convexnya

    /// FITUR DASAR - luas kontur, keliling kontur, luas convex, keliling convex

    double konturArea = contourArea(contours);
    double konturPerimeter = arcLength(contours, true);
    double konveksArea = contourArea(hull);
    double konveksPerimeter = arcLength(hull, true);

    ///-- roundness - tingkat kebulatan
    kebulatan = ((4*3.14*konturArea)/ konveksPerimeter);
    printf("roundness : %lf\n", kebulatan);

    ///-- solidity - tingkat kepadatan
    kepadatan = konturArea/konveksArea;
    printf("solidity : %lf\n", kepadatan);

    ///-- compactness - rasio antara objek dengan daerah lingkaran
    compactness = ((4*3.14*konturArea)/ pow(konturPerimeter,2));
    printf("compactness : %lf\n", compactness);

    ///-- convexity - jumlah relatif objek berbeda dari convex hull
    convexity = konveksPerimeter / konturPerimeter;
    printf("convexity : %lf\n", convexity);

    ///////////////////////////////////////////////
}

void dilasi(Mat gambar, int ukuran){
    Mat element = getStructuringElement(MORPH_DILATE,
                                        Size(2 * ukuran + 1, 2 * ukuran + 1),
                                        Point(-1,-1));
    dilate(gambar, gambar, element);
}

void erosi(Mat gambar, int ukuran){
    Mat element = getStructuringElement(MORPH_DILATE,
                                        Size(2 * ukuran + 1, 2 * ukuran + 1),
                                        Point(-1,-1));
    erode(gambar, gambar, element);
}

void cleanNoise(Mat gambar){

    GaussianBlur(gambar,gambar, Size(9,9), 0, 0);

    threshold(gambar,gambar, 57, 255, 0);

    GaussianBlur(gambar,gambar, Size(9,9), 0, 0);

    threshold(gambar,gambar, 57, 255, 0);

    GaussianBlur(gambar,gambar, Size(9,9), 0, 0);

    threshold(gambar,gambar, 57, 255, 0);

    GaussianBlur(gambar,gambar, Size(9,9), 0, 0);

    threshold(gambar,gambar, 57, 255, 0);
}


void segmentasi(Mat img){

    Mat channel[3];
    Mat subb ;

    split(img, channel);

    subtract(channel[2],channel[1],subb);

    imshow("Original", img);

//    imshow("Hasil sub", subb);

    threshold(subb, edge, 57, 175, 0);

//    imshow("hasil Threshold", edge);

    cleanNoise(edge);

//    imshow("Hasil cleaner", edge);

//    dilate(edgeMat, edgeMat, Mat(), Point(-1,-1), 6, 1, 1);
    dilasi(edge, 2);

//    imshow("Hasil dilasi", edge);

    erosi(edge, 5);

//    imshow("Hasil erosi", edge);

    src_gray = edge;

}

void cariKontur(Mat img, vector<double>* fitur, int status)
{
  int largestArea=0;
  int largestIndex=0;
  vector<Vec4i> hierarchy;
  vector<vector<Point> > contours;
  /// Find contours
  findContours( img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

//  for( int i = 0; i < contours.size(); i++ )
//     {
//            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
//            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
//            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
//     }

      if(contours.size() != 0){
        for( int i = 0; i< contours.size(); i++ ){
               double a=contourArea( contours[i],false);  //  Find the area of contour
               if(a>largestArea){
                largestArea=a;
                largestIndex=i;
               }
        }

                ekstrakFitur(contours[largestIndex]);

                fitur->push_back(kebulatan); //contour area
                fitur->push_back(kepadatan); // contour perimeter
                fitur->push_back(compactness); // convex area
                fitur->push_back(convexity); // convex perimter
                fitur->push_back(status);
      }
     }

void trainingData(int status){
    int N = 0;
    string path;
    if (status){
        N = 86; //jumlah data training class ikan
        path = pathDataTrainIkan;
    } else {
        N=11; //jumlah data training class non-ikan
        path = pathDataTrainNonIkan;
    }

    for(int i=1; i<=N;i++){
        string read = path+number_to_string(i)+".jpg";
        cout << i << endl;
        imgOri= imread(read);
        vector<double> fiturExtraction;
        fiturExtraction.push_back(i);
        segmentasi(imgOri);
        cariKontur(edge,&fiturExtraction,status);
        dataset.push_back(fiturExtraction);
    }
}

bool getKnn(vector<double>* feature){
    vector<vector<double> > knn;
    vector<double>& data = * feature;
    for(int i=0; i<dataset.size(); i++){
        vector<double> temp;
        double jarak = sqrt(pow((dataset[i][1]-data[0]),2)+pow((dataset[i][2]-data[1]),2)+pow((dataset[i][3]-data[2]),2)+pow((dataset[i][4]-data[3]),2));
        temp.push_back(jarak);
        temp.push_back(dataset[i][5]);
        knn.push_back(temp);
    }

    vector<double> hasilSort;
    sort(knn.begin(),knn.end());
    int ikan=0;
    int bukanIkan=0;
    for(int i=0;i<3;i++){
        if(knn[i][1]) ikan++; else bukanIkan++;
        //printf("jarak = %lf,kelas = %lf\n",knn[i][0],knn[i][1]);
    }
    if(ikan>bukanIkan) return true; else return false;
}

void testingData(Mat img){
    bool hasil;
    segmentasi(img);
    imshow("test", edge);
    /// Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    cout << "jumlah kontur" << contours.size() <<endl;
    if(contours.size() != 0){
        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
         {
                approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
                boundRect[i] = boundingRect( Mat(contours_poly[i]) );
                minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
         }



            for( int i = 0; i< contours.size(); i++ )
            {
//                if(contourArea(contours[i])>200){
                    vector<double> feature;

                    ekstrakFitur(contours[i]);

                    feature.push_back(kebulatan);
                    feature.push_back(kepadatan);
                    feature.push_back(compactness);
                    feature.push_back(convexity);

                    if(getKnn(&feature)){
                      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                      //drawContours( imgOri, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
                      rectangle( img, boundRect[i].tl(), boundRect[i].br(), color, 3, 8, 0 );
                      circle( imgOri, center[i], (int)radius[i], color, 3, 8, 0 );
                    }
//                }
             }
    }

      /// Show in a window
      namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
      imshow( "Contours", img );
}

int main()
{
    Mat testImg = imread("neon8.jpg");
    trainingData(1);
    trainingData(0);
    testingData(testImg);
    for(int i =0; i< dataset.size(); i++){
        for (int j =0; j <6 ; j++){
            cout << dataset[i][j] <<" ";
        }
        cout << endl;
    }
    cvWaitKey(0);


    return 0;
}
