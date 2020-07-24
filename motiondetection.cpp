#include <iostream>    
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/bgsegm.hpp>  
#include <opencv2/cudabgsegm.hpp>  
#include <opencv2/cudaoptflow.hpp>  

using namespace cv;
using namespace std;

static void download(const cuda::GpuMat &d_mat, vector<Point2f> &vec);
static void download(const cuda::GpuMat &d_mat, vector<uchar> &vec);
static void drawArrows(Mat &frame, const vector<Point2f> &prevPts, const vector<Point2f> &nextPts, const vector<uchar> &status, Scalar line_color = Scalar(0, 0, 255));


int main()
{
  //variable
  cuda::GpuMat GpuImg, rGpuImg_Bgray;
  cuda::GpuMat oldGpuImg_Agray;

  //video
  Mat img, dImg_rg, dimg;
  VideoCapture cap("./realtime_test.mp4");
  cap >> img;
  if (img.empty()) {
    cerr << "Video is empty!!!\n";
    exit(-1);
  }
  cout << "Start" << endl;

  //save init
  Size size = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
                (int)cap.get(CAP_PROP_FRAME_HEIGHT));
  VideoWriter writer;
  double fps = 30.0;
  writer.open("output.mp4", VideoWriter::fourcc('m', 'p', 'e', 'g'), fps, size, true);
  if (!writer.isOpened())
  {
    cout << "Failed to initailize save video" << endl;
    return 1;
  }


  //scale
  double scale = 800. / img.cols;

  //first gpumat
  cout << "Upload image to gpu" << endl;
  GpuImg.upload(img);
  cuda::resize(GpuImg, oldGpuImg_Agray, Size(GpuImg.cols * scale, GpuImg.rows * scale));
  cuda::cvtColor(oldGpuImg_Agray, oldGpuImg_Agray, cv::COLOR_BGR2GRAY);

  cuda::GpuMat d_prevPts;
  cuda::GpuMat d_nextPts;
  cuda::GpuMat d_status;
  Ptr< cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(oldGpuImg_Agray.type(), 4000, 0.01, 0);
  //optical flow
  Ptr< cuda::SparsePyrLKOpticalFlow> d_pyrLK = cuda::SparsePyrLKOpticalFlow::create(Size(21, 21), 3, 30);

  unsigned long Atime, Btime;
  float TakeTime;

  cout << "Loop start" << endl;
  while (1)
  {
    Atime = getTickCount();

    cap >> img;
    if (img.empty()) {
      cerr << "Video is empty!!!\n";
      break;
    }

    //get image
    //cout << "get image" << endl;
    GpuImg.upload(img);
    cuda::resize(GpuImg, rGpuImg_Bgray, Size(GpuImg.cols * scale, GpuImg.rows * scale));
    rGpuImg_Bgray.download(dimg);
    cuda::cvtColor(rGpuImg_Bgray, rGpuImg_Bgray, cv::COLOR_BGR2GRAY);
    rGpuImg_Bgray.download(dImg_rg);

    //A,B image  
    //oldGpuImg_Agray;
    //rGpuImg_Bgray;

    //feature
    cout << "detect feature" << endl;
    detector->detect(oldGpuImg_Agray, d_prevPts);
    d_pyrLK->calc(oldGpuImg_Agray, rGpuImg_Bgray, d_prevPts, d_nextPts, d_status);


    //old
    oldGpuImg_Agray = rGpuImg_Bgray;
    

    //Draw arrows
    //cout << "Draw arrows" << endl;
    vector< Point2f> prevPts(d_prevPts.cols);
    download(d_prevPts, prevPts);

    vector< Point2f> nextPts(d_nextPts.cols);
    download(d_nextPts, nextPts);

    vector< uchar> status(d_status.cols);
    download(d_status, status);

    drawArrows(dimg, prevPts, nextPts, status, Scalar(255, 0, 0));

    //save
    //cout << "Write video" << endl;
    writer.write(dimg);


    //show
    imshow("PyrLK [Sparse]", dimg);  
    //imshow("origin", dImg_rg);
    if (waitKey(10)>0)
      break;


    Btime = getTickCount();
    TakeTime = (Btime - Atime) / getTickFrequency();  
    printf("%lf sec / %lf fps \n", TakeTime, 1 / TakeTime);
    
    }
  return 0;
}



static void download(const cuda::GpuMat &d_mat, vector<uchar> &vec)
{
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
  d_mat.download(mat);
}

static void download(const cuda::GpuMat&d_mat, vector<Point2f>&vec)
{
  vec.resize(d_mat.cols);
  Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
  d_mat.download(mat);
}

static void drawArrows(Mat &frame, const vector<Point2f> &prevPts, const vector<Point2f> &nextPts, const vector<uchar> &status, Scalar line_color)
{
  int box_w1 = 0;
  int box_h1 = 0;
  int box_w2 = 0;
  int box_h2 = 0;
  for (size_t i = 0; i < prevPts.size(); ++i)
  {
    if (status[i])
    {
      int line_thickness = 1;

      Point p = prevPts[i];
      Point q = nextPts[i];

      double angle = atan2((double)p.y - q.y, (double)p.x - q.x);

      double hypotenuse = sqrt((double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x));

      if (hypotenuse < 1.0)
        continue;
      
      if (box_w1 == 0) {
        box_w1 = int(p.x);
      }
      if (box_h1 == 0) {
        box_h1 = int(p.y);
      }
      if (box_w1 > int(p.x)) {
        box_w1 = int(p.x);
      }
      if (box_w1 > int(p.y)) {
        box_h1 = int (p.y);
      }
      if (box_w2 < int(p.x)) {
        box_w2 = int(p.x);
      }
      if (box_h2 < int(p.y)) {
        box_h2 = int(p.y);
      }

      // Here we lengthen the arrow by a factor of three.
      q.x = (int)(p.x - 3 * hypotenuse * cos(angle));
      q.y = (int)(p.y - 3 * hypotenuse * sin(angle));

      // Now we draw the main line of the arrow.
      line(frame, p, q, line_color, line_thickness);

      // Now draw the tips of the arrow. I do some scaling so that the
      // tips look proportional to the main line of the arrow.

      p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
      p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
      line(frame, p, q, line_color, line_thickness);

      p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
      p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
      line(frame, p, q, line_color, line_thickness);
    }
  }
  // draw Object
  rectangle(frame, Point(box_w1, box_h1), Point(box_w2, box_h2), Scalar(0,255,0), 3);
}
